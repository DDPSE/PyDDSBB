#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Thu Jun  4 18:27:16 2020
@Updated on Tue Jul 08 01:15:48 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

import numpy as np
class Var:
    def __init__(self, lb, ub, vartype):
        self._bounds = np.array([lb, ub])
        self._vartype = vartype 
    def _get_vartype(self):
        return self._vartype
    def _get_bound(self):
        return self._bounds
    
class Problem:
    """
    Define the blackbox optimization problem with known bounds on the input space
    """
    def __init__(self):
        self._dim = 0
        self._number_known_constraint = 0
        self._number_unknown_constraint = 0
        self._variable = []
        self._type = []
    def add_variable(self, lb, ub, vartype = 'continuous'):
        self._dim += 1
        self._variable.append(Var(lb, ub, vartype))
        if vartype not in self._type:
            self._type.append(vartype)
    def add_objective(self, objective, sense = 'minimize'):
        """
        Add objective function 
        """
        self._objective = objective 
        self._sense = sense
    def add_known_constraint(self, known_constraint):
        """
        Add known constraint
        """
        self._number_known_constraint += 1
        setattr(self, '_known_constraint' + str(self._number_known_constraint), known_constraint)
    def update_sense(self, sense):
        """
        Update the sense of optimization
        """
        self._sense = sense
    def add_unknown_constraint(self, unknown_constraint):
        """
        Add unkown_constraints
        """
        self._number_unknown_constraint += 1
        setattr(self, '_unknown_constraint' + str(self._number_unknown_constraint), unknown_constraint)