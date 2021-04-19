#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:31:40 2020

@author: JianyuanZhai
"""

class Node:
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
        self.child.append(node)
    def add_parent(self, pn):
        self.pn = pn
    def add_data(self, x, y):
        self.x = x
        self.y = y
    def add_score(self, score):
        self.score = score
    def set_opt_local(self, fub, xopt):
        self.yopt_local = fub
        self.xopt_local = xopt
    def set_opt_flb(self, flb):
        self.flb = flb
    def set_decision(self, decision):
        self.decision = decision
    def print_node(self):
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
        self.label = label
    def add_valid_ind(self, valid_ind):
        self.valid_ind = valid_ind