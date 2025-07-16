#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Tue Jul  7 13:31:40 2020
@Updated on Tue Jul 08 01:15:48 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

class Node:
    """
    Node object contains information about lower bound, parent information
    """
    def __init__(self, level, node, bounds, pn = None):        
        self.node = node
        self.level = level
        self.bounds = bounds
        self.xrange = (self.bounds[1,:] - self.bounds[0,:])  
        self.min_xrange = max(self.xrange)
        self.decision = 1
        self.child = []
        self.pn = pn
    def add_child(self, node):
        """
        Add a child node number to this node for tracking
        Inputs
        ------
        node: int
              child node number    
        """
        self.child.append(node)
    def add_parent(self, node):
        """
        Add parent node number to this node
        Inputs
        ------
        node: int
              parent node number
        """
        self.pn = node
    def add_data(self, x, y):
        """
        Add data to this node
        Inputs
        ------
        x: ndarray of size (n_samples, n_variables)
           input data
        y: ndarray of size (n_samples, 1)
           output data
        """
        self.x = x
        self.y = y
    def add_score(self, score):
        """
        Add SVR or other variable ranking scores
        Inputs
        ------
        score: 1-D array of size (n_variables)
               score of each variables
        """
        self.score = score
    def set_opt_local(self, fub, xopt):
        """
        Set local optimum 
        Inputs
        ------
        fub: float
             local upper bound 
        xopt: float
             local minizer 
        """
        self.yopt_local = fub
        self.xopt_local = xopt
    def set_opt_flb(self, flb):
        """
        Set local lower bound
        Inputs
        ------
        flb: float 
             local lower bound 
        """
        self.flb = flb
    def set_lipschitz(self, lipschitz):
        self.lipschitz = lipschitz
    def set_decision(self, decision):
        """
        Set pruning decision 
        Inputs
        ------
        decision: int
                  1: keep the node active
                  0: prune the node
        """
        self.decision = decision
    def print_node(self):
        """
        Print node information
        """
        print('Node level:   ' + str(self.level))
        print('Node number:  ' + str(self.node))
        print('Local upper bound:  ' + str(self.yopt_local))
        print('Local lower bound:  ' + str(self.flb))
        print('Local gap: ' + str(self.yopt_local - self.flb))
        if self.decision == 0:
            print('Node pruned')
        else:
            print('Node active')
    def add_label(self, label):
        """
        Add feasibility label to the samples
        Inputs
        ------
        label: ndarray of size (n_samples, 1)
               label to each sample
               1: feasible sample
               0: infeasible sample
        """
        self.label = label
    def add_valid_ind(self, valid_ind):
        """
        Add indices to non-infinity samples in current sample set
        Inputs
        ------
        valid_ind: 1d- list of size (n_valid_sample)
                   valid sample indices
        """
        self.valid_ind = valid_ind