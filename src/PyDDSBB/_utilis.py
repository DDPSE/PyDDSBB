#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Thu Jun  4 21:02:17 2020
@Updated on Tue Jul 08 01:15:48 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

import numpy as np
class LHS:
    @staticmethod 
    def initial_sample(dim, number_new_points):
        '''
        input: required number of points (int); 
        return: new LHS points
        '''        
        current_design = np.array([[np.random.uniform(0, 1) for i in range(dim)]])
        new_design = LHS.augmentLHS(current_design, number_new_points - 1)
        return np.vstack((current_design,new_design))
    @staticmethod
    def augmentLHS(original_design, number_new_points): 
        '''
        input: a LHS design (type : np.array) and required number of points (int); 
        return: new LHS points
        '''        
        number_old_points, dim = original_design.shape
       
        ## Generate cells for LHS desgin ##
        number_cells = (number_old_points + number_new_points)**2
        cell_size = 1./(number_cells + 1)
        cell_lo = [i*cell_size for i in range(number_cells + 1)]
        cell_up = [(i + 1)*cell_size for i in range(number_cells + 1)]
        number_candidate_points = (number_new_points) * 2
        
        ## Find filled cells ##
        def find_filled(design):
            filtered = filter(lambda x : x >= 0, design - cell_lo)
            return len(list(filtered)) - 1
        
        ## return randomized design in unfilled cells ##
        def select_candidates(col_vec):
            allfilled = list(map(find_filled, col_vec))   
            candidate_cells = np.random.choice([k for k in range(number_cells + 1) if k not in allfilled],number_candidate_points)
            # return [float(np.random.uniform(cell_lo[k],cell_up[k],1)) for k in candidate_cells]
            return [float(np.random.uniform(cell_lo[k], cell_up[k])) for k in candidate_cells]


       
        candidate_points = [select_candidates(original_design[:,i]) for i in range(dim)]
        candidate_points_filtered = np.array(candidate_points).T          
        new_points = [] 
        current_design = original_design.copy()
        candidates = candidate_points_filtered.copy()       
        
        distance = np.min(np.sum(np.square(current_design[:, None] - candidates),2),0)
        
        ### Adaptively add new points with max-min-distance to existing LHS desging ###
        while len(new_points) != number_new_points:            
            selected = np.argmax(distance)            
            new_points.append(candidates[selected,:])            
            current_design = np.concatenate((original_design, new_points),axis=0)
            candidates = np.delete(candidates, selected, 0)  
            distance = np.delete(distance, selected, 0)            
            distance = np.array([min(np.sum((np.array(current_design[-1,:]) - candidates[i,:])**2),distance[i]) for i in range(len(candidates))])            
        new_design = np.array(new_points)
        
        
        return new_design
