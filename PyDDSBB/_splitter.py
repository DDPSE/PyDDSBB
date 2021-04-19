#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:34:48 2020

@author: JianyuanZhai
"""
import numpy as np
from _machine_learning import LocalSVR

class Splitter:
    def __init__(self, split_method, variable_selection = 'longest_side', minimum_bd = 0.05):
        self.minimum_bd = minimum_bd
        if split_method not in ['gini', 'purity']: 
            if variable_selection not in  ['gini', 'purity']:
                self.split_spot = getattr(self, split_method)
                self.split_x = getattr(self, variable_selection)
                self.split = self.split_reg
            
        else:            
            if variable_selection in [ 'gini', 'purity']:
                self.split = self.location_based
                self.criteria = getattr(self, variable_selection)
            else:
                self.split_x = getattr(self, variable_selection)
                self.split_spot = getattr(self, split_method)
                self.split = self.hybrid
    def split_reg(self, parent):       
        split_x = self.split_x(parent)
        split_spot = self.split_spot(parent, split_x)
        child_bound1 = parent.bounds.copy()
        child_bound2 = parent.bounds.copy()
        child_bound1[0, split_x] = split_spot
        child_bound2[1, split_x] = split_spot        
        return child_bound1, child_bound2 
    def hybrid(self, parent):
        split_x = self.split_x(parent)
        
        if sum(parent.label) != len(parent.label) and sum(parent.label) != 0.:
            split_spot, score = self.split_spot(parent, split_x)
        else:
            split_spot = Splitter.equal_bisection(parent, split_x)
        child_bound1 = parent.bounds.copy()
        child_bound2 = parent.bounds.copy()
        child_bound1[0, split_x] = split_spot
        child_bound2[1, split_x] = split_spot 
        return child_bound1, child_bound2 
    def location_based(self, parent):
        best_loc = []
        best_score = []
        if sum(parent.label) != len(parent.label) and sum(parent.label) != 0.:
            for i in range(parent.x.shape[1]):
                loc, score = self.criteria(parent, i)
                best_loc.append(loc)
                best_score.append(score)
            if best_score.count(2.) == parent.x.shape[1]:
                split_x = Splitter.longest_side(parent)
                split_spot = Splitter.equal_bisection(parent, split_x)
            else:
                split_x = best_score.index(min(best_score))      
                split_spot = best_loc[split_x]
        else:
            split_x = Splitter.longest_side(parent)
            split_spot = Splitter.equal_bisection(parent, split_x)
        child_bound1 = parent.bounds.copy()
        child_bound2 = parent.bounds.copy()
        child_bound1[0, split_x] = split_spot
        child_bound2[1, split_x] = split_spot        
        return child_bound1, child_bound2
    @staticmethod
    def equal_bisection(parent,split_x):        
        return 0.5*(parent.bounds[1, split_x] - parent.bounds[0, split_x]) + parent.bounds[0, split_x]
    @staticmethod
    def longest_side(parent):
        return np.argmax(parent.bounds[1, :] - parent.bounds[0, :])
    def purity(self, parent, split_x):
        if parent.xrange[split_x] >= self.minimum_bd:
            try:
                n_split = parent.x.shape[0]
                n_feasible = sum(parent.label)
                step = parent.xrange[split_x]/n_split
                cur_xup = parent.bounds[0, split_x] + step
                locations = {}
                while cur_xup < parent.bounds[1, split_x]:
                    if cur_xup - parent.bounds[0, split_x] >= self.minimum_bd/2 and parent.bounds[1, split_x] - cur_xup >= self.minimum_bd/2:
                        ind = [i for i in range(n_split) if parent.x[i, split_x] <= cur_xup]                    
                        if ind != [] and len(ind) < n_split: 
                            label = parent.label[ind]
                            p_left = (len(label) - sum(label))/(n_split - n_feasible) - sum(label)/n_feasible
                            locations[cur_xup] = max(p_left, - p_left)
                        else:
                            break
                    cur_xup += step
            
                best_location = max(locations, key = locations.get)
                return best_location, - locations[best_location]
            except:
                return None, 2.
        else:
            return None, 2.            
    def svr_var_select(self, parent):   
        if hasattr(parent, 'score') is False:
            X = (parent.x[parent.valid_ind, :] - parent.bounds[0,:]) / parent.xrange
            Y = (parent.y[parent.valid_ind] - min(parent.y[parent.valid_ind])) / (max(parent.y[parent.valid_ind]) - min(parent.y[parent.valid_ind]))
            MF = LocalSVR()
            MF._train(X, Y)
            parent.add_score(MF._rank())
        rank = np.argsort(parent.score)
        for i in rank:
            if parent.xrange[i] > self.minimum_bd:
                return i
        return Splitter.longest_side(parent)
    def gini(self, parent, split_x):
        if parent.xrange[split_x] >= self.minimum_bd:
            try:
                n_split = parent.x.shape[0]
                step = parent.xrange[split_x]/n_split
                cur_xup = parent.bounds[0, split_x] + step
                locations = {}
                while cur_xup < parent.bounds[1, split_x]:
                    if cur_xup - parent.bounds[0, split_x] >= self.minimum_bd/2. and parent.bounds[1, split_x] - cur_xup >= self.minimum_bd/2.:
                        ind = [i for i in range(n_split) if parent.x[i, split_x] <= cur_xup]                       
                        if ind != []:                           
                            ind2 = [i for i in range(n_split) if i not in ind]                            
                            if ind2 != []:
                                p_left = len(ind)/n_split
                                p_right = len(ind2)/n_split
                                p_feasible_left = sum(parent.label[ind])/len(ind)
                                p_feasible_right = sum(parent.label[ind2])/len(ind2)
                                locations[cur_xup] =  p_left*p_feasible_left*(1 - p_feasible_left)*2 + p_right*p_feasible_right*(1 - p_feasible_right)*2
                            else:
                                break
                    cur_xup += step    
                best_location = min(locations, key = locations.get)
                return best_location, locations[best_location]
            except:
                return None, 2.
        else:
            return None, 2.                