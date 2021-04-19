#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:39:07 2020

@author: JianyuanZhai
"""
import pyomo.environ as pe
import numpy as np
import time

DOUBLE = np.float64
class DDCU_Nonuniform():
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform(intercept)
        self.solver = pe.SolverFactory('glpk')  
        self.time_underestimate = 0.          
    @staticmethod
    def _minimize_1d(a, b, c):
        if a > 0.:
            check = -b/(2*a)
            if check >= 0. and check <= 1.:
                return check
            elif check < 0.:
                return 0.
            elif check > 1.:
                return 1.
        elif 0. >= a >= -10.** -5:
            if b > 0.:
                return 0.
            if b < 0.:
                return 1.
            else:
                return 0.5          
    def update_solver(self, solver, option = {}):
        self.solver = pe.SolverFactory(solver)
    def _underestimate(self, all_X, all_Y):
        time_start = time.time()
        dim = all_X.shape[1]
        sample_ind = list(range(len(all_Y)))
        x_ind = list(range(dim))
        x_dict = {}
        for i in sample_ind:
            for j in x_ind:
                x_dict[(i,j)] = all_X[i,j]   
        if self.intercept:
            data = {None:{'x_ind' : {None : x_ind} , 'sample_ind' : { None : sample_ind }, 'xs' : x_dict , 'ys' : dict(zip(sample_ind, all_Y))}}
        else:
            corner_point_ind = np.where((all_X == 0.).all(axis = 1))[0]            
            if len(corner_point_ind) > 1:
                candidate = all_X[corner_point_ind]                
                intercept = min(candidate)
            else:
                intercept = float(all_Y[corner_point_ind])     
            data = {None:{'x_ind' : {None : x_ind} , 'sample_ind' : { None : sample_ind } ,'xs' : x_dict , 'ys' : dict(zip(sample_ind, all_Y)) , 'c' : {None : intercept}}}
        model = self.ddcu.create_instance(data) # create an instance for abstract pyomo model
        self.solver.solve(model)              
        a = np.array([round(pe.value(model.a[i]), 6) for i in model.x_ind])
        if (a < 0.).any():
            model.pprint()
        b = np.array([pe.value(model.b[i]) for i in model.x_ind])
        c = pe.value( model.c )
        xopt = np.array([self._minimize_1d(a[i], b[i], c, ) for i in range(dim)])
        flb_s = sum(a*xopt**2+b*xopt) + c
        if abs(flb_s - min(all_Y)) <= 0.00001:
            flb_s = min(all_Y)
        self.time_underestimate += time.time() - time_start
        return float(flb_s), np.array([xopt])




class DDCU_model:
    """
    This class contains recipes to make pyomo models for different pyomo_models for underestimators
    """   
    @staticmethod
    def _linear_obj_rule(model):
        return sum((model.ys[i] - model.f[i]) for i in model.sample_ind)
    @staticmethod
    def _underestimating_con_rule(model, i):
        return model.ys[i] - model.f[i] >= 0.0
    @staticmethod
    def _quadratic_nonuniform(model, i):
        return model.f[i] == sum(model.a[j]*model.xs[i,j]**2 + model.b[j]*model.xs[i,j] for j in model.x_ind) + model.c
    @staticmethod
    def _exponential(model, i):
        a = sum(model.a[j]*(model.xs[i,j]-model.b[j])**2 for j in model.x_ind)
        return model.f[i] == pe.exp(a) + model.c
    @staticmethod
    def _make_pyomo_ddcu_nonuniform(intercept):       
        ddcu = pe.AbstractModel()
        ddcu.sample_ind = pe.Set()
        ddcu.x_ind = pe.Set()  
        ddcu.ys = pe.Param(ddcu.sample_ind)
        ddcu.xs = pe.Param(ddcu.sample_ind,ddcu.x_ind)
        ddcu.a = pe.Var(ddcu.x_ind,within = pe.NonNegativeReals, initialize=0.)
        ddcu.b = pe.Var(ddcu.x_ind,within = pe.Reals)               
        ddcu.f = pe.Var(ddcu.sample_ind)
        if intercept :
            ddcu.c = pe.Var(within = pe.Reals)
        else :
            ddcu.c = pe.Param()                
        ddcu.obj = pe.Objective(rule = DDCU_model._linear_obj_rule)
        ddcu.con1 = pe.Constraint(ddcu.sample_ind, rule = DDCU_model._underestimating_con_rule)
        ddcu.con2 = pe.Constraint(ddcu.sample_ind, rule = DDCU_model._quadratic_nonuniform)     
        return ddcu
    @staticmethod
    def _make_pyomo_ddcu_exponential():       
        ddcu = pe.AbstractModel()
        ddcu.sample_ind = pe.Set()
        ddcu.x_ind = pe.Set()  
        ddcu.ys = pe.Param(ddcu.sample_ind)
        ddcu.xs = pe.Param(ddcu.sample_ind,ddcu.x_ind)
        ddcu.a = pe.Var(ddcu.x_ind,within = pe.NonNegativeReals)
        #ddcu.b = pe.Param(ddcu.x_ind) 
        ddcu.b = pe.Var(ddcu.x_ind, within = pe.Reals)              
        ddcu.f = pe.Var(ddcu.sample_ind)
        ddcu.c = pe.Var(within = pe.Reals)
        ddcu.obj = pe.Objective(rule = DDCU_model._linear_obj_rule)
        ddcu.con1 = pe.Constraint(ddcu.sample_ind, rule = DDCU_model._underestimating_con_rule)
        ddcu.con2 = pe.Constraint(ddcu.sample_ind, rule = DDCU_model._exponential)     
        return ddcu
