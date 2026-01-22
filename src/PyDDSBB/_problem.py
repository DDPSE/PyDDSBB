#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Thu Jun  4 18:15:08 2020
@Updated on Tue Jul 08 01:15:48 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

import numpy as np
import time
class BoundConstrainedSimulation:
    problem_type = 'BoundConstrained'
    def __init__(self, problem):
        self._dim = problem._dim
        self._bounds = np.array([i._bounds for i in problem._variable]).T
        self._objective = problem._objective
        self._sense = problem._sense
        self.time_sampling = 0.
        if self._sense == 'minimize':
            self._simulate = self._obj_minimize
        elif self._sense == 'maximize':
            self._simulate = self._obj_maximize   
        self.sample_number = 0
    def _obj_maximize(self, x):   
        time_start = time.time()
        self.sample_number += len(x)
        y = np.array([-self._objective(x[j,:]) for j in range(len(x))])
        self.time_sampling += time.time() - time_start
        if len(y) == 1:
            return y.item()
        else:
            return y        
    def _obj_minimize(self,x):
        time_start = time.time()
        self.sample_number += len(x)
        y = np.array([self._objective(x[j,:]) for j in range(len(x))])
        self.time_sampling += time.time() - time_start
        if len(y) == 1:
            return y.item()
        else:
            return y
    
    
class BlackBoxSimulation(BoundConstrainedSimulation):
    problem_type = 'BlackBox'
    def __init__(self, problem):
        super().__init__(problem)
        self._number_unknown_constraint = problem._number_unknown_constraint
        self._unknown_constraints = []
        self.time_constraints = 0
        for i in range(problem._number_unknown_constraint):
            self._unknown_constraints.append(getattr(problem, '_unknown_constraint' + str(i + 1)))
    def _check_feasibility(self, x):  
        time_start = time.time()
        n = x.shape[0]
        feasibility = np.zeros(n)      
        for j in range(n):
            feasibility_x =[constr(x[j,:]) for constr in self._unknown_constraints]
            if sum(feasibility_x) == self._number_unknown_constraint:
                feasibility[j] = 1. 
        self.time_constraints += time.time() - time_start
        return feasibility 
    
class GreyBoxSimulation(BlackBoxSimulation):
    problem_type = 'GreyBox'
    def __init__(self, problem):
        super().__init__(problem)
        self._number_unknown_constraint = problem._number_unknown_constraint
        self._number_known_constraint = problem._number_known_constraint
        self._unknown_constraints = []
        self._known_constraints = []
        self.time_constraints = 0
        for i in range(problem._number_unknown_constraint):
            self._unknown_constraints.append(getattr(problem, '_unknown_constraint' + str(i + 1)))
        for i in range(problem._number_known_constraint):
            self._known_constraints.append(getattr(problem, '_known_constraint' + str(i + 1)) + ' + eps' + str(i + 1))
        
