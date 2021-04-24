#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyDDSBB @ GT - DDPSE

@author: JianyuanZhai
"""
import numpy as np
from PyDDSBB._utilis import LHS
import PyDDSBB._problem as _problem 
import PyDDSBB._underestimators 
import time
from PyDDSBB._node import Node
from PyDDSBB._splitter import Splitter
from PyDDSBB._machine_learning import LocalSVR
import pyomo.environ as pe

UNDERESTIMATORS = {'Quadratic': PyDDSBB._underestimators.DDCU_Nonuniform}
INFINITY = np.inf


class Tree:
    def __init__(self):                  
        self.Tree = {}  
        self.current_level = 0 
        self.Tree[self.current_level] = {}
        self.flb_current = INFINITY
        self.yopt_global = INFINITY
        self.xopt_global = None
        self.min_xrange = INFINITY
    def _activate_node(self):
        pass  
    def _add_level(self):    
        self.current_level += 1
        self.Tree[self.current_level] = {}         
        self.lowerbound_global = self.flb_current 
        self.flb_current = INFINITY  
        self._xopt_hist.append(self.xopt_global)
    def _add_node(self, node):
        if node.yopt_local <= self.yopt_global:
            self.yopt_global = node.yopt_local
            self.best_node = node.node
            self.best_level = node.level
            self.xopt_global = node.xopt_local        
        if node.flb > self.yopt_global: 
            node.set_decision(0)
        else:
            if node.yopt_local == INFINITY: 
                if node.level == 1:
                    if self.Tree[node.level - 1][node.pn].yopt_local == INFINITY:
                        node.set_decision(0)
                if node.level > 1:
                    parent = self.Tree[node.level - 1][node.pn]
                    if parent.yopt_local == INFINITY and self.Tree[parent.level - 1][parent.pn].yopt_local == INFINITY:
                        node.set_decision(0)
            else:
                node.set_decision(1) 
                if node.flb < self.flb_current:
                    self.flb_current = node.flb
                if node.min_xrange < self.min_xrange:
                    self.min_xrange = node.min_xrange
        self.Tree[self.current_level][node.node] = node 
        
   
    
class NodeOperation:
    """
    Parent class for all node operation
    """
    def __init__(self, multifidelity, split_method, variable_selection, underestimator_option, minimum_bd):
        """
        Inputs
        ------
        multifidelity:  bool
                        True to turn on multifidelity option 
                        False to turn off multifidelity option
        split_method: str
        variable_selection: str
        underestimator_option: str
        minimum_bd: float 
        """
        self._underestimate = UNDERESTIMATORS[underestimator_option]()._underestimate        
        self.multifidelity = multifidelity 
        self.split = Splitter(split_method, variable_selection, minimum_bd).split  
        self.variable_selection = variable_selection
        if multifidelity is not False or self.variable_selection == 'svr_var_select':
            self.MF = LocalSVR()        
        self.time_underestimate = 0.
    def _set_adaptive(self, adaptive_number):
        """
        Set adaptive sampling rule
        Input
        -----
        adaptive_number: int
        """
        self.adaptive_number = adaptive_number 
    def _adaptive_sample(self):
        """
        Use augmented latin hypercube strategy to add more samples
        """
        x_corner = np.zeros((2,self.dim))
        x_corner[1,:] = 1.0       
        self._update_sample(x_corner)
        if self.adaptive_number - len(self.y) > 0:
            Xnew = LHS.augmentLHS(self.X, self.adaptive_number - len(self.y))
            self._update_sample(Xnew)
         ## Check if cornor points the samples already, if not sample them
    def _min_max_rescaler(self, Xnew):
        """
        Scale Xnew by the original bounds 
        Input
        ------
        Xnew: ndarry of shape (n_samples, n_variables)
        
        Return
        ------
        xnew: ndarray of shape (n_samples, n_variables)
        """
        xnew = Xnew*self.xrange + self.bounds[0, :]
        return xnew     
    def _split_node(self, parent):
        """
        Split a node into two child node apply split method
        Input
        -----
        parent: node 
        
        Returns
        -------
        child1, child2: node 
        """
        child_bound1, child_bound2 = self.split(parent) 
        child1 = self._create_child(child_bound1, parent) 
        child2 = self._create_child(child_bound2, parent)
        return child1, child2   
    
    
    
class BoxConstrained(NodeOperation):
    """
    Node operations for box-constrained problems
    Derived class from NodeOperation
    """
    def __init__(self, multifidelity, split_method, variable_selection, underestimator_option, minimum_bd):
        super().__init__(multifidelity, split_method, variable_selection, underestimator_option, minimum_bd)         
    def _add_problem(self, problem):
        """
        Add problem to node operator
        Input
        -----
        problem: DDSBBModel
        """
        self.simulator = _problem.BoundConstrainedSimulation(problem)
        self.bounds = self.simulator._bounds 
        self.dim = self.simulator._dim        
    def _min_max_single_scaler(self):
        """
        Scale one sample between 0 and 1 based on the variable bounds and range of y
        """
        self.ymin_local = float(self.y)
        self.ymax_local = float(self.y)        
        self.xrange = (self.bounds[1, :] - self.bounds[0, :]) 
        self.X = (self.x - self.bounds[0, :])/self.xrange   
        if self.valid_ind != []:
            self.yopt_local = float(self.y)
            self.xopt_local = self.x
        else:
            self.yopt_local = INFINITY
            self.xopt_local = None
        self.yrange = self.ymax_local - self.ymin_local 
        if self.yrange== 0. :
            self.Y = 1.
        else:
            self.Y = (self.y - self.ymin_local)/ self.yrange 
    def _min_max_scaler(self):    
        """
        Scale current samples between 0 and 1 based on the variable bounds and range of y
        """            
        if self.valid_ind != []:
            self.yopt_local = min(self.y[self.valid_ind])
            min_ind = np.where(self.y == self.yopt_local)
            self.xopt_local = self.x[min_ind]
            self.ymin_local = min(self.y[self.valid_ind])        
            self.ymax_local = max(self.y[self.valid_ind])
        else:
            self.yopt_local = INFINITY
            self.xopt_local = None
            self.ymin_local = min(self.y)        
            self.ymax_local = max(self.y)
        self.yrange = self.ymax_local - self.ymin_local 
        self.xrange = self.bounds[1, :] - self.bounds[0, :]
        if self.yrange== 0. :
            self.Y = np.ones(self.y.shape)
        else:
            self.Y = (self.y - self.ymin_local)/ self.yrange  
        self.X = (self.x - self.bounds[0, :])/self.xrange 
    def _create_child(self, child_bounds, parent):
        """
        create a child node 
        
        Inputs
        ------
        child_bounds: ndarray of shape (2, n_variables)
                      bounds of the search space of the child node 
                      lower bound in row 1
                      upper bound in row 2
        parent: node 
                parent node 
        
        Return
        ------
        child: node
               child node with added samples, LB and UB informations
        """
        self.level = parent.level + 1
        ind1 = np.where((parent.x <= child_bounds[1, :]).all(axis=1) == True)
        ind2 = np.where((parent.x >= child_bounds[0, :]).all(axis=1) == True)  
        ind  = np.intersect1d(ind1,ind2) 
        self.x = parent.x[ind, :]
        self.y = parent.y[ind] 
        self.valid_ind = [i for i in range(len(ind)) if self.y[i] != INFINITY]        
        self.bounds = child_bounds
        self._min_max_scaler()
        self._adaptive_sample()
        flb = self._training_DDCU()    
        self.node += 1
        child = Node(parent.level + 1, self.node, self.bounds, parent.node)
        child.add_data(self.x, self.y)
        child.set_opt_flb(flb)
        child.set_opt_local(self.yopt_local, self.xopt_local)
        if self.variable_selection == 'svr_var_selection':
            child.add_score(self.MF.rank())
        child.add_valid_ind(self.valid_ind)
        return child
    def _update_sample(self, Xnew):
        """
        Update current sample set with new samples Xnew
        Input
        -----
        Xnew: ndarray of shape (n_samples, n_variables)
              new samples scaled between 0 and 1
        
        """
        index = [i for i in range(len(Xnew)) if (np.round(abs(self.X - Xnew[i, :]), 3) != 0.).all()] 
        if index != []:
            Xnew = Xnew[index, :]
            xnew = self._min_max_rescaler(Xnew)
            ynew = self.simulator._simulate(xnew) 
            self.X = np.concatenate((self.X, Xnew), axis=0)
            self.x = np.concatenate((self.x, xnew), axis=0)             
            if type(ynew) is float:
                if ynew == -INFINITY:
                    raise TypeError('ERROR: Problem Unbounded')
                else:
                    if ynew != INFINITY:
                        self.valid_ind += [len(self.y)]
                    self.y = np.append(self.y, ynew) 
                    if ynew < self.yopt_local:
                        self.yopt_local = float(ynew)
                        self.xopt_local = xnew
                    if  ynew >= self.ymin_local and ynew <= self.ymax_local:     
                        Ynew = (ynew-self.ymin_local)/self.yrange               
                        self.Y = np.append(self.Y, Ynew)   
                    elif ynew > self.ymax_local:
                        if ynew != INFINITY:
                            self.ymax_local = float(ynew)
                        self.yrange = self.ymax_local - self.ymin_local 
                        self.Y = (self.y - self.ymin_local)/self.yrange
                    elif ynew < self.ymin_local:                        
                        self.ymin_local = float(ynew)
                        self.yrange = self.ymax_local - ynew                                
                        self.Y = (self.y - ynew)/self.yrange
            else:
                min_ind = np.argmin(ynew)
                ymin = float(ynew[min_ind])
                if ymin == -INFINITY:
                    raise TypeError('ERROR: Problem Unbounded')
                else:
                    ymax = max(ynew)                
                    current = len(self.y)                
                    valid_ind = [i for i in range(len(ynew)) if ynew[i] != INFINITY]   
                    if valid_ind != []:
                        ymin_feasible = min(ynew[valid_ind]) 
                        if ymin_feasible < self.yopt_local:
                            min_ind_feasible = [i for i in range(len(ynew)) if ynew[i] == ymin_feasible][0]
                            self.yopt_local = ymin_feasible
                            self.xopt_local = np.array([xnew[min_ind_feasible, :]])
                        self.valid_ind += [i + current for i in valid_ind]
                    self.y = np.append(self.y, ynew)                 
                    if ymin >= self.ymin_local and ymax <= self.ymax_local: 
                        Ynew = (ynew - self.ymin_local)/self.yrange                
                        self.Y = np.append(self.Y, Ynew)                              
                    elif ymin >= self.ymin_local and ymax > self.ymax_local:
                        if ymax != INFINITY:
                            self.ymax_local = ymax
                        self.yrange = self.ymax_local - self.ymin_local
                        self.Y = (self.y - self.ymin_local)/self.yrange
                    elif ymin < self.ymin_local and ymax <= self.ymax_local:     
                        self.ymin_local = ymin
                        self.yrange = self.ymax_local - self.ymin_local                                 
                        self.Y = (self.y - self.ymin_local)/self.yrange
                    elif ymin < self.ymin_local and ymax > self.ymax_local:
                        self.ymin_local = ymin
                        if ymax != INFINITY:
                            self.ymax_local = ymax
                        self.yrange = self.ymax_local - self.ymin_local
                        self.Y = (self.y - self.ymin_local)/self.yrange 
    def _create_root_node(self): 
        """
        Create root node for the BB tree
        """
        self.level = 0        
        self.x = self.bounds
        self.node = 0
        self.y = self.simulator._simulate(self.bounds)
        self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
        self._min_max_scaler()        
        self._adaptive_sample()
        flb = self._training_DDCU()
        root_node = Node(self.level, self.node, self.bounds)        
        root_node.add_data(self.x, self.y)
        root_node.set_opt_flb(flb)
        root_node.set_opt_local(self.yopt_local, self.xopt_local)
        if self.variable_selection == 'svr_var_selection':
            root_node.add_score(self.MF.rank())
        root_node.add_valid_ind(self.valid_ind)
        return root_node  
    def _training_DDCU(self):
        """
        Train DDCU 
        
        Return
        ------
        flb: float 
             lower bound 
        """
        time_start = time.time()
        iteration = 0
        flb = INFINITY
        while flb > self.ymin_local :
            check = 0            
            while True:
                try:
                    if self.multifidelity is False:
                        all_X = self.X[self.valid_ind, :]
                        all_Y = self.Y[self.valid_ind]
                    elif iteration == 0 :
                        self.MF._train(self.X[self.valid_ind,:], self.Y[self.valid_ind])                        
                        lowfidelity_X = np.random.random_sample((min(50*self.dim, 251), self.dim))            
                        lowfidelity_Y = self.MF._predict(lowfidelity_X)
                        lowfidelity_Xopt = lowfidelity_X[np.argmin(lowfidelity_Y), :]
                        self._update_sample(np.array([lowfidelity_Xopt]))
                        all_Y = np.concatenate((self.Y[self.valid_ind], lowfidelity_Y), axis=0)
                        all_X = np.concatenate((self.X[self.valid_ind, :], lowfidelity_X), 0)
                    flb_s, Xnew = self._underestimate(all_X, all_Y)
                    break
                except:
                    check += 1
                    if check > 20:
                        raise RuntimeError('ERROR: Failed to train DDCU')
            flb = flb_s * self.yrange + self.ymin_local
            self._update_sample(Xnew)
            if abs(self.ymin_local - flb) <= 1E-6:
                flb = self.ymin_local
            iteration += 1
            if iteration > 20:
                raise RuntimeError('ERROR: Failed to find a valid bound within 20 iterations')
        self.time_underestimate += time.time() - time_start

        return float(flb)

class BlackBox(NodeOperation):
    """
    Node operation for pure black-box constrained problems
    """
    def __init__(self, multifidelity, split_method, variable_selection, underestimator_option, sampling_limit, minimum_bd):
        super().__init__(multifidelity , split_method , variable_selection , underestimator_option, minimum_bd)
        self.sampling_limit = sampling_limit
    def _add_problem(self, problem):
        """
        Add problem to node operator
        Input
        -----
        problem: DDSBBModel
        """
        self.simulator = _problem.BlackBoxSimulation(problem)
        self.bounds = self.simulator._bounds 
        self.dim = self.simulator._dim  
    def _min_max_single_scaler(self):
        """
        Scale one sample between 0 and 1 based on the variable bounds and range of y
        """
        self.ymin_local = float(self.y)
        self.ymax_local = float(self.y)        
        self.xrange = (self.bounds[1, :] - self.bounds[0, :]) 
        self.X = (self.x - self.bounds[0, :])/self.xrange   
        if self.feasible_ind != []:
            self.yopt_local = float(self.y)
            self.xopt_local = self.x
        else:
            self.yopt_local = INFINITY
            self.xopt_local = None
        self.yrange = self.ymax_local - self.ymin_local 
        if self.yrange== 0. :
            self.Y = 1.
        else:
            self.Y = (self.y - self.ymin_local)/ self.yrange       
    def _min_max_scaler(self):
        """
        Scale current samples between 0 and 1 based on the variable bounds and range of y
        """ 
        self.ymin_local = min(self.y)        
        self.ymax_local = max(self.y)
        if self.feasible_ind != []:
            self.yopt_local = min(self.y[self.feasible_ind])
            min_ind = np.where(self.y == self.yopt_local)
            self.xopt_local = self.x[min_ind]
        else:
            self.yopt_local = INFINITY
            self.xopt_local = None
        self.yrange = self.ymax_local - self.ymin_local 
        self.xrange = self.bounds[1, :] - self.bounds[0, :]
        if self.yrange == 0. :
            self.Y = np.ones(self.y.shape)
        else:
            self.Y = (self.y - self.ymin_local)/ self.yrange  
        self.X = (self.x - self.bounds[0, :])/self.xrange   
    def _update_sample(self, Xnew):
        """
        Update current sample set with new samples Xnew
        Input
        -----
        Xnew: ndarray of shape (n_samples, n_variables)
              new samples scaled between 0 and 1
        
        """
        index = [i for i in range(len(Xnew)) if (np.round(abs(self.X - Xnew[i, :]), 3) != 0.).all()] ## Find which samples are to be collected 
        if index != []:
            Xnew = Xnew[index, :]
            xnew = self._min_max_rescaler(Xnew)
            ynew = self.simulator._simulate(xnew)        
            label = self.simulator._check_feasibility(xnew)            
            self.X = np.concatenate((self.X, Xnew), axis=0)
            self.x = np.concatenate((self.x, xnew), axis=0) 
            self.label = np.append(self.label, label)
            if type(ynew) is float:    ### Only One Sample                         
                if ynew == -INFINITY:  ### If sample infeasible 
                    raise TypeError('ERROR: Problem Unbounded')
                else: ### if sample is feasible
                    if ynew != INFINITY:
                        self.valid_ind += [len(self.y)]
                    self.y = np.append(self.y, ynew)
                    if ynew < self.yopt_local and label[0] == 1.:                    
                        self.yopt_local = float(ynew)
                        self.xopt_local = xnew
                        self.feasible_ind += [len(self.y)]
                    if  ynew >= self.ymin_local and ynew <= self.ymax_local:     
                        Ynew = (ynew-self.ymin_local)/self.yrange               
                        self.Y = np.append(self.Y, Ynew)   
                    elif ynew > self.ymax_local:
                        if ynew != INFINITY:
                            self.ymax_local = float(ynew)
                            self.yrange = self.ymax_local - self.ymin_local 
                            self.Y = (self.y - self.ymin_local)/self.yrange
                    elif ynew < self.ymin_local:
                        self.ymin_local = float(ynew)
                        self.yrange = self.ymax_local - ynew                                
                        self.Y = (self.y - ynew)/self.yrange
            else: ### Add multiple samples  
                min_ind = np.argmin(ynew)
                ymin = float(ynew[min_ind])
                if ymin == -INFINITY: ### If sample minimum is infeasible 
                    raise TypeError('ERROR: Problem Unbounded')
                else:     ### If sample sample minimmum is feasible 
                    ymax = max(ynew)                
                    current = len(self.y)                
                    feasible = [i for i in range(len(label)) if label[i] == 1.]   
                    valid_ind = [current + i for i in range(len(ynew)) if ynew[i] != INFINITY]
                    if valid_ind != []:
                        self.valid_ind += valid_ind
                    if feasible != []:
                        ymin_feasible = min(ynew[feasible]) 
                        if ymin_feasible < self.yopt_local:
                            min_ind_feasible = [i for i in range(len(ynew)) if ynew[i] == ymin_feasible][0]
                            self.yopt_local = ymin_feasible
                            self.xopt_local = np.array([xnew[min_ind_feasible, :]])
                        self.feasible_ind += [i + current for i in feasible]
                    self.y = np.append(self.y, ynew)                 
                    if ymin >= self.ymin_local and ymax <= self.ymax_local: 
                        Ynew = (ynew - self.ymin_local)/self.yrange                
                        self.Y = np.append(self.Y, Ynew)                              
                    elif ymin >= self.ymin_local and ymax > self.ymax_local:
                        if ymax != INFINITY:
                            self.ymax_local = ymax
                        self.yrange = self.ymax_local - self.ymin_local
                        self.Y = (self.y - self.ymin_local)/self.yrange
                    elif ymin < self.ymin_local and ymax <= self.ymax_local:                    
                        self.ymin_local = ymin
                        self.yrange = self.ymax_local - self.ymin_local                                 
                        self.Y = (self.y - self.ymin_local)/self.yrange                    
                    elif ymin < self.ymin_local and ymax > self.ymax_local:
                        self.ymin_local = ymin
                        if ymax != INFINITY:
                            self.ymax_local = ymax
                        self.yrange = self.ymax_local - self.ymin_local
                        self.Y = (self.y - self.ymin_local)/self.yrange 

    def _create_child(self, child_bounds, parent): 
        """
        create a child node 
        
        Inputs
        ------
        child_bounds: ndarray of shape (2, n_variables)
                      bounds of the search space of the child node 
                      lower bound in row 1
                      upper bound in row 2
        parent: node 
                parent node 
        
        Return
        ------
        child: node
               child node with added samples, LB and UB informations
        """
        self.level = parent.level + 1
        ind1 = np.where((parent.x <= child_bounds[1, :] + 10**(-10)).all(axis=1) == True)
        ind2 = np.where((parent.x >= child_bounds[0, :] - 10**(-10)).all(axis=1) == True) 
        ind  = np.intersect1d(ind1,ind2)  
        self.x = parent.x[ind, :]
        self.y = parent.y[ind]         
        self.label = parent.label[ind]        
        self.feasible_ind = [i for i in range(len(self.label)) if self.label[i] == 1.]
        self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
        self.bounds = child_bounds
        if len(self.y) > 1:            
            self._min_max_scaler()
        elif len(self.y) == 1: 
            self._min_max_single_scaler() 
        self._adaptive_sample()
        flb = self._training_DDCU()    
        self.node += 1
        child = Node(parent.level + 1, self.node, self.bounds, parent.node)
        child.add_data(self.x, self.y)
        child.set_opt_flb(flb)
        child.set_opt_local(self.yopt_local, self.xopt_local)
        child.add_label(self.label)
        if self.variable_selection == 'svr_var_selection':
            child.add_score(self.MF.rank())
        child.add_valid_ind(self.valid_ind)
        return child
    def _create_root_node(self): 
        """
        Create root node for the BB tree
        """
        self.level = 0        
        self.x = self.bounds
        self.node = 0
        self.y = self.simulator._simulate(self.bounds)
        self.label = self.simulator._check_feasibility(self.x)
        self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
        self.feasible_ind = [i for i in range(len(self.label)) if self.label[i] == 1.]        
        self._min_max_scaler()   
        self._adaptive_sample()
        while self.feasible_ind == [] and self.simulator.sample_number < self.sampling_limit:
            self._set_adaptive(self.simulator.sample_number + 11*self.dim + 1)
            self._adaptive_sample()        
        flb = self._training_DDCU()        
        root_node = Node(self.level, self.node, self.bounds)        
        root_node.add_data(self.x, self.y)
        root_node.set_opt_flb(flb)
        root_node.add_label(self.label)
        root_node.set_opt_local(self.yopt_local, self.xopt_local)
        if self.variable_selection == 'svr_var_selection':
            root_node.add_score(self.MF.rank())
        root_node.add_valid_ind(self.valid_ind)
        return root_node  
    def _training_DDCU(self):
        """
        Train DDCU 
        
        Return
        ------
        flb: float 
             lower bound 
        """
        time_start = time.time()
        iteration = 0
        flb = INFINITY
        while flb > self.ymin_local :
            check = 0            
            while True:
                try:
                    if self.multifidelity is False:
                        all_X = self.X[self.valid_ind, :]
                        all_Y = self.Y[self.valid_ind]
                    elif iteration == 0 :
                        self.MF._train(self.X[self.valid_ind, :], self.Y[self.valid_ind])                        
                        lowfidelity_X = np.random.random_sample((min(50*self.dim, 251), self.dim))            
                        lowfidelity_Y = self.MF._predict(lowfidelity_X)
                        lowfidelity_Xopt = lowfidelity_X[np.argmin(lowfidelity_Y), :]
                        self._update_sample(np.array([lowfidelity_Xopt]))
                        all_Y = np.concatenate((self.Y[self.valid_ind], lowfidelity_Y), axis=0)
                        all_X = np.concatenate((self.X[self.valid_ind, :], lowfidelity_X), 0)                   
                    flb_s, Xnew = self._underestimate(all_X, all_Y)
                    break
                except:
                    check += 1
                    if check > 20:
                        raise 'FAILDED'
            flb = flb_s * self.yrange + self.ymin_local
            self._update_sample(Xnew)
            if abs(self.ymin_local - flb) <= 10**-6:
                flb = self.ymin_local
            iteration +=1
            if iteration > 20:
                raise "ERROR: Failed to find a valid bound within 20 iterations"
        self.time_underestimate += time.time() - time_start
        return float(flb)
        
class GreyBox(BlackBox):
    """
    Node operations for greybox constrained problems (mixed know and unknown problems)
    """
    def __init__(self, multifidelity, split_method, variable_selection, underestimator_option, sampling_limit, minimum_bd):
        super().__init__(multifidelity , split_method , variable_selection , underestimator_option, sampling_limit, minimum_bd)
    def _add_problem(self, problem):
        """
        Add problem to the solver
        
        Input
        -----
        problem: DDSBBModel
        """
        self.simulator = _problem.GreyBoxSimulation(problem)
        self.bounds = self.simulator._bounds 
        self.dim = self.simulator._dim 
        self.model = pe.ConcreteModel()  ## Initialize a model for FBBT
        for i in range(self.dim):
            setattr(self.model, 'x' + str(i), pe.Var(within = pe.Reals))  
            setattr(self.model, 'obj' + str(i), pe.Objective(expr = getattr(self.model, 'x' + str(i)))) ## Regular objective for FBBT on root node
            getattr(self.model, 'obj' + str(i)).deactivate() ## deactivate objective function for FBBT 
            
            
        obj = [] ### Objective for feasibility check by adding slack variable eps (constraint violation)
        for j in range(self.simulator._number_known_constraint): ## iteratively adding slack variabel eps_j to each constraint 
            setattr(self.model, 'eps' + str(j + 1), pe.Var(within = pe.NonNegativeReals))
            cons = self.simulator._known_constraints[j]
            for i in range(self.dim):
                cons = cons.replace('x' + str(i), 'self.model.x' + str(i))
            cons = cons.replace('eps', 'self.model.eps')
            cons = cons.replace('np', 'pe')
            setattr(self.model, 'con' + str(j + 1), pe.Constraint(expr = eval(cons)))
            obj.append('self.model.eps' + str(j + 1)) ### Sum of slack varible 
        
        obj = '+'.join(obj)
        self.model.obj_eps = pe.Objective(expr = eval(obj))
        self.model.obj_eps.deactivate() ## Deactivate objective function for feasibility check 
        self.solver = pe.SolverFactory('ipopt')
    def _check_node_feasibility(self):  
        """
        Solve feasibility check problem at each node
        
        Return
        ------
        1 for feasible node
        0 for infeasible node 
        """    
        self.model.obj_eps.activate()
        ind = 0
        for i in  self.model.component_objects(pe.Var, active = True):
            if 'x' in str(i):
                i.setlb(self.bounds[0,ind])  ### Update lower bound on variable 
                i.setub(self.bounds[1,ind])  ### Update upper bound on variable
            ind += 1        
        self.solver.solve(self.model)
        con_violation = pe.value(self.model.obj_eps) ## get sum of constraint violation
        self.model.obj_eps.deactivate() ## Deactivate objective 
        if con_violation == 0.:
            return 1 
        else:
            return 0
    def _bound_tighten(self):
        """
        Solve feasibility-based bound tightening at root node

        """
        for j in range(self.simulator._number_known_constraint):
            getattr(self.model, 'eps' + str(j + 1)).fix(0.)
            
        for i in range(self.dim):
            getattr(self.model, 'obj' + str(i)).activate()
            getattr(self.model, 'x' + str(i)).value = self.bounds[0, i]
            self.solver.solve(self.model)
            self.bounds[0, i] = pe.value(getattr(self.model, 'obj' + str(i)))
            getattr(self.model, 'obj' + str(i)).set_sense(-1)
            getattr(self.model, 'x' + str(i)).value = self.bounds[1, i]
            self.solver.solve(self.model)
            self.bounds[1, i] = pe.value(getattr(self.model, 'obj' + str(i)))
            getattr(self.model, 'obj' + str(i)).deactivate()
        
        for j in range(self.simulator._number_known_constraint):
            getattr(self.model, 'eps' + str(j + 1)).unfix()
        print('xlb after FBBT:  ' + str(self.bounds[0, :]))
        print('xub after FBBT:  ' + str(self.bounds[1, :]))
    def _create_child(self, child_bounds, parent):
        """
        create a child node 
        
        Inputs
        ------
        child_bounds: ndarray of shape (2, n_variables)
                      bounds of the search space of the child node 
                      lower bound in row 1
                      upper bound in row 2
        parent: node 
                parent node 
        
        Return
        ------
        child: node
               child node with added samples, LB and UB informations
        """
        self.bounds = child_bounds
        self.level = parent.level + 1
        if (child_bounds == np.nan).any():
            raise ValueError()
        ind1 = np.where((parent.x <= child_bounds[1, :]).all(axis=1) == True)
        ind2 = np.where((parent.x >= child_bounds[0, :]).all(axis=1) == True)  
        ind  = np.intersect1d(ind1,ind2)  
        self.x = parent.x[ind, :].copy()
        self.y = parent.y[ind].copy()         
        self.label = parent.label[ind].copy() 
        check = self._check_node_feasibility() ### Check feasibility of the node before adding more samples and underestimating 
        self.node += 1
        child = Node(parent.level + 1, self.node, self.bounds, parent.node)
        if  check == 1:                           
            self.feasible_ind = [i for i in range(len(self.label)) if self.label[i] == 1.]  
            self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
            if len(ind) > 1:            
                self._min_max_scaler()
            else:
                self.x = parent.x[ind, :]
                self._min_max_single_scaler()                        
            self._adaptive_sample()
            flb = self._training_DDCU()                
            child.add_data(self.x, self.y)
            child.set_opt_flb(flb)
            child.set_opt_local(self.yopt_local, self.xopt_local)
            child.add_label(self.label)
            if self.variable_selection == 'svr_var_selection':
                child.add_score(self.MF.rank())
            child.add_valid_ind(self.valid_ind)
        elif check == 0:            
            child.set_opt_flb(INFINITY)
            child.add_data(self.x, self.y)
            child.add_label(self.label)
            child.set_opt_local(INFINITY,  None)
        return child
    def _create_root_node(self): 
        """
        Create root node for the BB tree
        """
        self.level = 0        
        self.x = self.bounds
        check = self._check_node_feasibility() ### Check feasibility of the root node 
        self.node = 0
        if check == 1:
            self._bound_tighten() ### tighten the bound 
            self.y = self.simulator._simulate(self.bounds)
            self.label = self.simulator._check_feasibility(self.x)
            self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
            self.feasible_ind = [i for i in range(len(self.label)) if self.label[i] == 1.]        
            self._min_max_scaler()   
            self._adaptive_sample()
            while self.feasible_ind == [] and self.simulator.sample_number < self.sampling_limit:
                self._set_adaptive(self.simulator.sample_number + 11*self.dim + 1)
                self._adaptive_sample()        
            flb = self._training_DDCU()        
            root_node = Node(self.level, self.node, self.bounds)        
            root_node.add_data(self.x, self.y)
            root_node.set_opt_flb(flb)
            root_node.add_label(self.label)
            root_node.set_opt_local(self.yopt_local, self.xopt_local)
            if self.variable_selection == 'svr_var_selection':
                root_node.add_score(self.MF.rank())
            root_node.add_valid_ind(self.valid_ind)
        elif check == 0:
            root_node = Node(self.level, self.node, self.bounds)
            root_node.set_opt_flb(INFINITY)
            root_node.set_opt_local(INFINITY, None)
        return root_node  

class DDSBB(Tree):
    def __init__(self, number_init_samples, multifidelity = False, split_method = 'equal_bisection', variable_selection = 'longest_side', underestimator_option = 'Quadratic', stop_option = {'absolute_tolerance': 0.05, 'relative_tolerance': 0.01, 'minimum_bound': 0.05, 'sampling_limit': 10000, 'time_limit': 36000}, sense = 'minimize', adaptive_sampling = None):
        """
        Initialize DDSBB solver
        
        Inputs
        ------
        number_init_samples: int
                            Number of initial samples
        multifidelity: bool
                       True to turn on multifidelity approach 
                       False to turn off multifidelity approach (default)
        split_method: str
                      select from: equal_bisection (default), golden_section (for all problems)
                                   purity, gini (constrained problems)
        variable_selection: str
                      select from: longest_side (default), svr_var_select (for all problem)
                                   purity, gini (constrained problems)
        underestimator_option: str
                      Default: Quadratic
        stop_option: dict
                    absolute_tolerance: float
                    relative_tolerance: float
                    minimum_bound: float
                    sampling_limit: int
                    time_limit: float
        sense: str
               select from: minimize, maximize 
        adaptive_sampling: function of level, dimension
        """
        super().__init__()
        self.current_level = 0   
        self.node = 0
        self.level = 0
        self.stop = 0
        self.sample_number = 0    
        
        
        self.init_sample = number_init_samples       
        self.stop_message = 'Method Initialized'
        self.multifidelity = multifidelity
        self.split_method = split_method
        self.variable_selection = variable_selection
        self.underestimator_option = underestimator_option
                           
        for i in stop_option.keys():
            setattr(self, i, stop_option[i])
        if sense == 'minimize' :
            self.report_LB = 'lower bound'
            self.report_UB = 'upper bound'
        else:
            self.report_LB = 'upper bound'
            self.report_UB = 'lower bound'   
        if adaptive_sampling is not None:
            self._adaptive = adaptive_sampling
    def print_result(self):
        """
        print solution status
        """
        print(self.stop_message)
        print("Time elapsed: " + str(round(self.time_total, 2)) + 's')
        print("Current level: " + str(self.level))
        print("Current node: " + str(self.builder.node))
        print("Number of samples used: " + str(self.builder.simulator.sample_number))
        print("Current best " + self.report_UB + " :  " + str(self.yopt_global))
        print("Current best " + self.report_LB + " :  " + str(self.lowerbound_global))
        print("Current absolute gap:  " + str(self.yopt_global - self.lowerbound_global))
        print("Current best optimizer: " + str(self.xopt_global))
    def update_stop_criteria(self, new_stop):
        """
        update stop criteria
        
        Input
        -----
        new_stop: dict
                 dictionary constaining new stopping criteria 
        """
        for i in new_stop.keys():
            setattr(self, i, new_stop[i])        
        self.stop = 0                                        
    def _grow(self):  
        """
        grow the tree 
        """
        while(self.stop == 0):
            self.level += 1
            self._add_level()
            self.builder._set_adaptive(self._adaptive_rule())     
            self._completion_indicator = False 
            for parent in self.Tree[self.level - 1].values(): 
                if parent.decision == 1:
                    if parent.flb <= self.yopt_global:
                        child1, child2 = self.builder._split_node(parent)
                        self._add_node(child1)
                        parent.add_child(child1.node)
                        self._add_node(child2)
                        parent.add_child(child2.node)
                    elif parent.flb > self.yopt_global:
                        parent.decision = 0
                self._check_resources()
                if self.stop != 0:
                    self.last_searchednode = parent.node 
                    break
            if self.stop == 0:
                self._completion_indicator = True
                self._check_convergence()
    def _continue(self):
        """
        continue search with new stopping criteria 
        """
        for parent in self.Tree[self.level - 1].values():            
            if parent.node > self.last_searchednode:
                if parent.decision == 1:
                    if parent.flb <= self.yopt_global:
                        child1, child2 = self.builder._split_node(parent)
                        self._add_node(child1)
                        parent.add_child(child1.node)
                        self._add_node(child2)
                        parent.add_child(child2.node)
                    elif parent.flb > self.yopt_global:
                        parent.decision = 0
                self._check_resources()
                if self.stop != 0:
                    self.last_searchednode = parent.node 
                    break 
        if self.stop == 0:
            self._completion_indicator = True 
            self._check_convergence()
    @staticmethod 
    def _adaptive(dim, level):
        """
        Default adaptive sampling rule
        """
        return max(int((dim)*11/level+3),int(dim*3)+3)
    def _adaptive_rule(self):
        """
        adaptive sampling rule 
        """
        return self._adaptive(self.dim, self.level)
    def optimize(self, problem):
        """
        Optimize the problem with DDSBB algorithm
        Input
        -----
        problem: DDSBBModel
        """
        
        
        self._lowerbound_hist = []
        self._upperbound_hist = []
        self._xopt_hist = []
        self._sampling_hist = []
        self._cpu_hist = []
        self.time_start = time.time()
        self.search_instance = 1
        self.stop_message = 'In search process'
        if problem._number_unknown_constraint == 0 and problem._number_known_constraint == 0:            
            self.builder = BoxConstrained(self.multifidelity, self.split_method, self.variable_selection, self.underestimator_option, self.minimum_bound)
        elif problem._number_unknown_constraint != 0  and problem._number_known_constraint == 0:
            self.builder = BlackBox(self.multifidelity, self.split_method, self.variable_selection, self.underestimator_option, self.sampling_limit, self.minimum_bound)
        elif problem._number_unknown_constraint != 0 and problem._number_known_constraint != 0:
            self.builder = GreyBox(self.multifidelity, self.split_method, self.variable_selection, self.underestimator_option, self.sampling_limit, self.minimum_bound)
        self.builder._add_problem(problem)
        self.builder._set_adaptive(self.init_sample)  
        self.dim = self.builder.dim
        self._add_node(self.builder._create_root_node())
        self._completion_indicator = True 
        self._check_convergence()
        if self.stop == 0.:
            self._check_resources()
        self._grow()
        self.time_total = time.time() - self.time_start
    def resume(self, new_stop_option):
        self.time_elapsed = self.time_total 
        self.time_start = time.time()
        self.search_instance += 1
        print('Resume search with new resources')
        self.stop_message = 'In search process ' + str(self.search_instance)
        self.update_stop_criteria(new_stop_option)
        if self._completion_indicator:
            self._grow()
        else:
            self._continue()
            if self._completion_indicator:
                self._grow()
        self.time_total = time.time() - self.time_start + self.time_elapsed
    def _check_convergence(self):
        self.lowerbound_global = self.flb_current
        self._lowerbound_hist.append(self.lowerbound_global)
        self._upperbound_hist.append(self.yopt_global)
        self._sampling_hist.append(self.builder.simulator.sample_number)
        if self.search_instance == 1:
            self.time_total = time.time() - self.time_start
        else:
            self.time_total = time.time() - self.time_start + self.time_elapsed
        self._cpu_hist.append(self.time_total)
        if self.yopt_global != INFINITY:
            if self.yopt_global - self.lowerbound_global <= self.absolute_tolerance:
                self.stop = 1
                self.stop_message = 'absolute gap closed'
            else:
                if self.lowerbound_global != 0.:
                    if (self.yopt_global - self.lowerbound_global)/abs(self.lowerbound_global) <= self.relative_tolerance:
                        self.stop = 2
                        self.stop_message = 'relative gap closed'
                    else:
                        if self.min_xrange <= self.minimum_bound:
                            self.stop = 3
                            self.stop_message = 'search space too small'
                else:
                    if self.min_xrange <= self.minimum_bound:
                        self.stop = 3
                        self.stop_message = 'search space too small'
        else:
            self.stop = 4
            self.stop_message = 'Problem Infeasible'
    def _check_resources(self):
        if self.search_instance == 1:
            self.time_total = time.time() - self.time_start
        else:
            self.time_total = time.time() - self.time_start + self.time_elapsed
        self.sample_number = self.builder.simulator.sample_number
        if self.sample_number >= self.sampling_limit:
            self.stop_message = 'reached sampling limit'
            self.stop = 4 
        else:
            if self.time_total >= self.time_limit:
                self.stop_message = 'reached time limit'
                self.stop = 5
        if self.stop != 0 and self._completion_indicator is False:
            self.stop_message = self.stop_message + ' without search all active nodes in level ' + str(self.level)


