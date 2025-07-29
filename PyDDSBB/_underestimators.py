#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Thu Jun  4 20:39:07 2020
@Updated on Tue Jul 28 12:25:28 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

import pyomo.environ as pe
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
import warnings


DOUBLE = np.float64

class DDCU_Nonuniform():
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform(intercept)
        self.solver = pe.SolverFactory('glpk')  
        self.time_underestimate = 0.          

    @staticmethod
    def _minimize_1d_vec(a_array, b_array, c):
        """Vectorized version of minimize function over [0,1] for quadratic."""
        xopt = np.zeros_like(a_array)
        with np.errstate(divide='ignore', invalid='ignore'):
            a_positive = a_array > 0.
            check = -b_array / (2 * a_array)
            within_bounds = (check >= 0.) & (check <= 1.)
            xopt[a_positive & within_bounds] = check[a_positive & within_bounds]
            xopt[a_positive & (check < 0.)] = 0.
            xopt[a_positive & (check > 1.)] = 1.

            a_zero = np.isclose(a_array, 0., atol=1e-5)
            xopt[a_zero & (b_array > 0.)] = 0.
            xopt[a_zero & (b_array < 0.)] = 1.
            xopt[a_zero & np.isclose(b_array, 0.)] = 0.5

        return xopt
        
    def update_solver(self, solver, option = {}):
        self.solver = pe.SolverFactory(solver)
    def _underestimate(self, all_X, all_Y, lowfidelity_X, lowfidelity_Y, xrange, yrange, xbounds, ymin_local, overallXrange):
        time_start = time.time()
        
        u, indices = np.unique(all_X, axis = 0, return_index=True)
        all_X = all_X[indices]
        all_Y = all_Y[indices]
        dim = all_X.shape[1]
        sample_ind = list(range(len(all_Y)))
        x_ind = list(range(dim))
        x_dict = {}
        for i in sample_ind:
            for j in x_ind:
                x_dict[(i,j)] = all_X[i,j]   
        if self.intercept:
            data = {None:{'x_ind' : {None : x_ind} , 'xs' : x_dict , 'ys' : dict(zip(sample_ind, all_Y)) ,'sample_ind' : { None : sample_ind }}}
        else:
            corner_point_ind = np.where((all_X == 0.).all(axis = 1))[0]            
            if len(corner_point_ind) > 1:
                candidate = all_X[corner_point_ind]                
                intercept = min(candidate)
            else:
                intercept = float(all_Y[corner_point_ind])     
            data = {None:{'x_ind' : {None : x_ind} , 'xs' : x_dict , 'ys' : dict(zip(sample_ind, all_Y)) ,'sample_ind' : { None : sample_ind } , 'c' : {None : intercept}}}
            
        model = self.ddcu.create_instance(data) # create an instance for abstract pyomo model
        self.solver.solve(model)              

        a = np.array([pe.value(model.a[i]) for i in model.x_ind])
        b = np.array([pe.value(model.b[i]) for i in model.x_ind])
        c = pe.value(model.c)
       
        TOLERANCE = 1e-5
        if any(val < -TOLERANCE for val in a):
            print("Negative 'a' coefficients found beyond the tolerance level:")
#             for j in model.x_ind:
#                 if model.a[j].value < -TOLERANCE:
#                     print(f"  a[{j}] = {model.a[j].value}")
#             model.pprint()
            raise ValueError("Negative 'a' coefficients found in the model solution.")

        
        xopt = self._minimize_1d_vec(a, b, c)
        flb_s = np.sum(a * xopt**2 + b * xopt) + c

        if abs(flb_s - np.min(all_Y)) <= 1e-5:
            flb_s = np.min(all_Y)

        self.time_underestimate += time.time() - time_start
        maxL = np.nan

       
        return float(flb_s), maxL, np.array([xopt])
   

  ##=======================================================================================================================

    

class DDCU_Nonuniform_with_LC():
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform_with_LC(intercept)
        self.time_underestimate = 0.

    @staticmethod
    def _minimize_1d_vec(a_array, b_array, c):
        """Vectorized version of minimize function over [0,1] for quadratic."""
        xopt = np.zeros_like(a_array)
        with np.errstate(divide='ignore', invalid='ignore'):
            a_positive = a_array > 0.
            check = -b_array / (2 * a_array)
            within_bounds = (check >= 0.) & (check <= 1.)
            xopt[a_positive & within_bounds] = check[a_positive & within_bounds]
            xopt[a_positive & (check < 0.)] = 0.
            xopt[a_positive & (check > 1.)] = 1.

            a_zero = np.isclose(a_array, 0., atol=1e-5)
            xopt[a_zero & (b_array > 0.)] = 0.
            xopt[a_zero & (b_array < 0.)] = 1.
            xopt[a_zero & np.isclose(b_array, 0.)] = 0.5

        return xopt

    def update_solver(self, solver, options=None):
        self.solver = pe.SolverFactory(solver)
        if options:
            for key, value in options.items():
                self.solver.options[key] = value
    
  
    
    @staticmethod
    def estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5):
        """
        Estimate the Lipschitz constant by considering the nearest neighbors.

        Parameters:
        - all_X: numpy.ndarray, shape (n_samples, n_features)
        - all_Y: numpy.ndarray, shape (n_samples,)
        - n_neighbors: int, number of nearest neighbors to consider

        Returns:
        - maxL: float, estimated Lipschitz constant
        - rates: dict, Lipschitz constants for each sample
        """
        n_samples = all_X.shape[0]

        if n_samples == 0:
            raise ValueError("The input dataset all_X is empty.")
        if n_samples == 1:
            # Only one sample; Lipschitz constant is zero
            return 0.0, {0: 0.0}

        max_possible_neighbors = n_samples - 1  # Exclude the point itself
        if n_neighbors > max_possible_neighbors:
            n_neighbors = max_possible_neighbors
            warnings.warn(
                f"n_neighbors is greater than the number of available samples minus one. "
                f"Adjusted n_neighbors to {n_neighbors}.",
                UserWarning
            )

        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(all_X)

        distances, indices = nbrs.kneighbors(all_X)

        # Exclude the first neighbor (the point itself)
        distances = distances[:, 1:]  # Shape: (n_samples, n_neighbors)
        indices = indices[:, 1:]      # Shape: (n_samples, n_neighbors)

        f_differences = np.abs(all_Y[:, np.newaxis] - all_Y[indices])  # Shape: (n_samples, n_neighbors)

        with np.errstate(divide='ignore', invalid='ignore'):
            lipschitz_ratios = np.divide(f_differences, distances)
            lipschitz_ratios = np.nan_to_num(lipschitz_ratios, nan=0.0, posinf=0.0, neginf=0.0)

        maxL = np.max(lipschitz_ratios)

        rates = {i: maxL for i in range(n_samples)}

        return maxL, rates

    def _underestimate(self, all_X, all_Y, lowfidelity_X, lowfidelity_Y, xrange, yrange, xbounds, ymin_local, overallXrange):
        time_start = time.time()
        


        _, indices = np.unique(all_X, axis=0, return_index=True)
        all_X = all_X[indices]
        all_Y = all_Y[indices]

        dim = all_X.shape[1]
        sample_ind = range(len(all_Y))
        x_ind = range(dim)


        x_dict = {(i, j): all_X[i, j] for i in sample_ind for j in x_ind}

        # Estimate Lipschitz constant using nearest neighbors
        maxL, rates = self.estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5)

        endpoints = [sample_ind[0], sample_ind[-1]]
        

        if self.intercept:
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'Rates': rates,
                    'endpoints': {None: endpoints}  
                }
            }
        else:
            corner_point_ind = np.where(np.all(np.isclose(all_X, 0.0), axis=1))[0]
            if len(corner_point_ind) == 0:
                raise ValueError("No corner point at origin found in all_X.")
            intercept = float(np.min(all_Y[corner_point_ind]))
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'c': {None: intercept},
                    'Rates': rates,
                    'endpoints': {None: endpoints}  
                }
            }


        self.solver = pe.SolverFactory('gurobi')
        self.solver.options['Presolve'] = 0 

        model = self.ddcu.create_instance(data)
        self.solver.solve(model)#, tee=True)

        a = np.array([pe.value(model.a[i]) for i in model.x_ind])
        b = np.array([pe.value(model.b[i]) for i in model.x_ind])
        c = pe.value(model.c)
       
        TOLERANCE = 1e-5
        if any(val < -TOLERANCE for val in a):
            print("Negative 'a' coefficients found beyond the tolerance level:")
#             for j in model.x_ind:
#                 if model.a[j].value < -TOLERANCE:
#                     print(f"  a[{j}] = {model.a[j].value}")
#             model.pprint()
            raise ValueError("Negative 'a' coefficients found in the model solution.")


        xopt = self._minimize_1d_vec(a, b, c)
        flb_s = np.sum(a * xopt**2 + b * xopt) + c

        if abs(flb_s - np.min(all_Y)) <= 1e-5:
            flb_s = np.min(all_Y)

        self.time_underestimate += time.time() - time_start
        return float(flb_s), maxL, np.array([xopt])
    
  ##=======================================================================================================================
    

class DDCU_Nonuniform_with_LC_and_IC():
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.time_underestimate = 0.

    @staticmethod
    def _minimize_1d_vec(a_array, b_array, c):
        """Vectorized version of minimize function over [0,1] for quadratic."""
        xopt = np.zeros_like(a_array)
        with np.errstate(divide='ignore', invalid='ignore'):
            a_positive = a_array > 0.
            check = -b_array / (2 * a_array)
            within_bounds = (check >= 0.) & (check <= 1.)
            xopt[a_positive & within_bounds] = check[a_positive & within_bounds]
            xopt[a_positive & (check < 0.)] = 0.
            xopt[a_positive & (check > 1.)] = 1.

            a_zero = np.isclose(a_array, 0., atol=1e-5)
            xopt[a_zero & (b_array > 0.)] = 0.
            xopt[a_zero & (b_array < 0.)] = 1.
            xopt[a_zero & np.isclose(b_array, 0.)] = 0.5

        return xopt

    def update_solver(self, solver, options=None):
        self.solver = pe.SolverFactory(solver)
        if options:
            for key, value in options.items():
                self.solver.options[key] = value
    
    
    @staticmethod
    def estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5):
        """
        Estimate the Lipschitz constant by considering the nearest neighbors.

        Parameters:
        - all_X: numpy.ndarray, shape (n_samples, n_features)
        - all_Y: numpy.ndarray, shape (n_samples,)
        - n_neighbors: int, number of nearest neighbors to consider

        Returns:
        - maxL: float, estimated Lipschitz constant
        - rates: dict, Lipschitz constants for each sample
        """
        n_samples = all_X.shape[0]

        if n_samples == 0:
            raise ValueError("The input dataset all_X is empty.")
        if n_samples == 1:
            # Only one sample; Lipschitz constant is zero
            return 0.0, {0: 0.0}

        max_possible_neighbors = n_samples - 1  # Exclude the point itself
        if n_neighbors > max_possible_neighbors:
            n_neighbors = max_possible_neighbors
            warnings.warn(
                f"n_neighbors is greater than the number of available samples minus one. "
                f"Adjusted n_neighbors to {n_neighbors}.",
                UserWarning
            )

        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(all_X)

        distances, indices = nbrs.kneighbors(all_X)

        # Exclude the first neighbor (the point itself)
        distances = distances[:, 1:]  # Shape: (n_samples, n_neighbors)
        indices = indices[:, 1:]      # Shape: (n_samples, n_neighbors)

        f_differences = np.abs(all_Y[:, np.newaxis] - all_Y[indices])  # Shape: (n_samples, n_neighbors)

        with np.errstate(divide='ignore', invalid='ignore'):
            lipschitz_ratios = np.divide(f_differences, distances)
            lipschitz_ratios = np.nan_to_num(lipschitz_ratios, nan=0.0, posinf=0.0, neginf=0.0)

        maxL = np.max(lipschitz_ratios)

        rates = {i: maxL for i in range(n_samples)}

        return maxL, rates

    def _underestimate(self, all_X, all_Y, lowfidelity_X, lowfidelity_Y, xrange, yrange, xbounds, ymin_local, overallXrange):
        time_start = time.time()

        _, indices = np.unique(all_X, axis=0, return_index=True)
        all_X = all_X[indices]
        all_Y = all_Y[indices]

        dim = all_X.shape[1]
        sample_ind = range(len(all_Y))
        x_ind = range(dim)

        x_dict = {(i, j): all_X[i, j] for i in sample_ind for j in x_ind}

        # Estimate Lipschitz constant using nearest neighbors
        maxL, rates = self.estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5)
        endpoints = [sample_ind[0], sample_ind[-1]]

        if self.intercept:
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'Rates': rates,
                    'endpoints': {None: endpoints}  
                }
            }
        else:
            corner_point_ind = np.where(np.all(np.isclose(all_X, 0.0), axis=1))[0]
            if len(corner_point_ind) == 0:
                raise ValueError("No corner point at origin found in all_X.")
            intercept = float(np.min(all_Y[corner_point_ind]))
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'c': {None: intercept},
                    'Rates': rates,
                    'endpoints': {None: endpoints}  
                }
            }
            
        IR = max(xrange/overallXrange)
                

        self.solver = pe.SolverFactory('gurobi')
        self.solver.options['Presolve'] = 0 


        if IR >= 0.1:
            self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform_with_LC(self.intercept)
        else: 
            self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform(self.intercept)


        model = self.ddcu.create_instance(data)
        self.solver.solve(model)

        a = np.array([pe.value(model.a[i]) for i in model.x_ind])
        b = np.array([pe.value(model.b[i]) for i in model.x_ind])
        c = pe.value(model.c)

        
        TOLERANCE = 1e-5
        if any(val < -TOLERANCE for val in a):
            print("Negative 'a' coefficients found beyond the tolerance level:")
#             for j in model.x_ind:
#                 if model.a[j].value < -TOLERANCE:
#                     print(f"  a[{j}] = {model.a[j].value}")
#             model.pprint()
            raise ValueError("Negative 'a' coefficients found in the model solution.")


        xopt = self._minimize_1d_vec(a, b, c)
        flb_s = np.sum(a * xopt**2 + b * xopt) + c

        if abs(flb_s - np.min(all_Y)) <= 1e-5:
            flb_s = np.min(all_Y)

        self.time_underestimate += time.time() - time_start

        return float(flb_s), maxL, np.array([xopt])    

  ##=======================================================================================================================


class DDCU_Nonuniform_with_LC_and_bound():
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.ddcu = DDCU_model._make_pyomo_ddcu_nonuniform_LC_with_bound(intercept)
        self.time_underestimate = 0.

    @staticmethod
    def _minimize_1d_vec(a_array, b_array, c):
        """Vectorized version of minimize function over [0,1] for quadratic."""
        xopt = np.zeros_like(a_array)
        with np.errstate(divide='ignore', invalid='ignore'):
            a_positive = a_array > 0.
            check = -b_array / (2 * a_array)
            within_bounds = (check >= 0.) & (check <= 1.)
            xopt[a_positive & within_bounds] = check[a_positive & within_bounds]
            xopt[a_positive & (check < 0.)] = 0.
            xopt[a_positive & (check > 1.)] = 1.

            a_zero = np.isclose(a_array, 0., atol=1e-5)
            xopt[a_zero & (b_array > 0.)] = 0.
            xopt[a_zero & (b_array < 0.)] = 1.
            xopt[a_zero & np.isclose(b_array, 0.)] = 0.5

        return xopt

    def update_solver(self, solver, options=None):
        self.solver = pe.SolverFactory(solver)
        if options:
            for key, value in options.items():
                self.solver.options[key] = value
    
  
    
    @staticmethod
    def estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5):
        """
        Estimate the Lipschitz constant by considering the nearest neighbors.

        Parameters:
        - all_X: numpy.ndarray, shape (n_samples, n_features)
        - all_Y: numpy.ndarray, shape (n_samples,)
        - n_neighbors: int, number of nearest neighbors to consider

        Returns:
        - maxL: float, estimated Lipschitz constant
        - rates: dict, Lipschitz constants for each sample
        """
        n_samples = all_X.shape[0]

        if n_samples == 0:
            raise ValueError("The input dataset all_X is empty.")
        if n_samples == 1:
            # Only one sample; Lipschitz constant is zero
            return 0.0, {0: 0.0}

        max_possible_neighbors = n_samples - 1  # Exclude the point itself
        if n_neighbors > max_possible_neighbors:
            n_neighbors = max_possible_neighbors
            warnings.warn(
                f"n_neighbors is greater than the number of available samples minus one. "
                f"Adjusted n_neighbors to {n_neighbors}.",
                UserWarning
            )

        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(all_X)

        distances, indices = nbrs.kneighbors(all_X)

        # Exclude the first neighbor (the point itself)
        distances = distances[:, 1:]  # Shape: (n_samples, n_neighbors)
        indices = indices[:, 1:]      # Shape: (n_samples, n_neighbors)

        f_differences = np.abs(all_Y[:, np.newaxis] - all_Y[indices])  # Shape: (n_samples, n_neighbors)

        with np.errstate(divide='ignore', invalid='ignore'):
            lipschitz_ratios = np.divide(f_differences, distances)
            lipschitz_ratios = np.nan_to_num(lipschitz_ratios, nan=0.0, posinf=0.0, neginf=0.0)

        maxL = np.max(lipschitz_ratios)
        rates = {i: maxL for i in range(n_samples)}

        return maxL, rates

    def _underestimate(self, all_X, all_Y, lowfidelity_X, lowfidelity_Y, xrange, yrange, xbounds, ymin_local, overallXrange):
        time_start = time.time()
        


        _, indices = np.unique(all_X, axis=0, return_index=True)
        all_X = all_X[indices]
        all_Y = all_Y[indices]

        dim = all_X.shape[1]
        sample_ind = range(len(all_Y))
        x_ind = range(dim)
        

        x_dict = {(i, j): all_X[i, j] for i in sample_ind for j in x_ind}

        # Estimate Lipschitz constant using nearest neighbors
        maxL, rates = self.estimate_lipschitz_constant(all_X, all_Y, n_neighbors=5)

        endpoints = [sample_ind[0], sample_ind[-1]]
        xdist = np.sqrt(sum((all_X[sample_ind[0]] - all_X[sample_ind[-1]]) ** 2 ))
        ysum = all_Y[sample_ind[0]] + all_Y[sample_ind[-1]]
        
        if self.intercept:
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'L': {None: maxL},
                    'endpoints': {None: endpoints},  
                    'ysum': {None: ysum}, 
                    'xdist': {None: xdist}  
                }
            }
        else:
            corner_point_ind = np.where(np.all(np.isclose(all_X, 0.0), axis=1))[0]
            if len(corner_point_ind) == 0:
                raise ValueError("No corner point at origin found in all_X.")
            intercept = float(np.min(all_Y[corner_point_ind]))
            data = {
                None: {
                    'x_ind': {None: list(x_ind)},
                    'xs': x_dict,
                    'ys': dict(zip(sample_ind, all_Y)),
                    'sample_ind': {None: list(sample_ind)},
                    'c': {None: intercept},
                    'L': {None: maxL},
                    'endpoints': {None: endpoints},  
                    'ysum': {None: ysum}, 
                    'xdist': {None: xdist}  
                }
            }


            
        self.solver = pe.SolverFactory('glpk')
        model = self.ddcu.create_instance(data)
        self.solver.solve(model)#, tee=True)

        a = np.array([pe.value(model.a[i]) for i in model.x_ind])
        b = np.array([pe.value(model.b[i]) for i in model.x_ind])
        c = pe.value(model.c)
       
        TOLERANCE = 1e-5
        if any(val < -TOLERANCE for val in a):
            print("Negative 'a' coefficients found beyond the tolerance level:")
#             for j in model.x_ind:
#                 if model.a[j].value < -TOLERANCE:
#                     print(f"  a[{j}] = {model.a[j].value}")
#             model.pprint()
            raise ValueError("Negative 'a' coefficients found in the model solution.")



        xopt = self._minimize_1d_vec(a, b, c)
        flb_s = np.sum(a * xopt**2 + b * xopt) + c

        if abs(flb_s - np.min(all_Y)) <= 1e-5:
            flb_s = np.min(all_Y)

        self.time_underestimate += time.time() - time_start
        return float(flb_s), maxL, np.array([xopt])


  ##=======================================================================================================================


class DDCU_model:
    """
    This class contains recipes to make pyomo models for different pyomo_models for underestimators
    """   
    @staticmethod
    def _linear_obj_rule(model):
        return sum((model.ys[i] - model.f[i]) for i in model.sample_ind)
  #---------------------------------------------------------------------
    
    @staticmethod
    def _underestimating_con_rule(model, i):
        return model.ys[i] - model.f[i] >= 0.0   
  #---------------------------------------------------------------------
    
    @staticmethod
    def _underestimating_lipschitz_pos_rule(model, i, j):
        """
        For each i and j, enforce:
        model.a[j] * model.xs[i,j] * 2 + model.b[j] >= model.Rates[i] 
        """
        return model.a[j] * model.xs[i, j] * 2 + model.b[j] >= model.Rates[i] 
  #---------------------------------------------------------------------

    @staticmethod
    def _underestimating_lipschitz_neg_rule(model, i, j):
        """
        For each i and j, enforce:
        model.a[j] * model.xs[i,j] * 2 + model.b[j] <= -model.Rates[i] 
        """
        return model.a[j] * model.xs[i, j] * 2 + model.b[j] <= -model.Rates[i] 
  #---------------------------------------------------------------------

    @staticmethod
    def _lipschitz_lb_on_underestimator_min_rule(model):

        return sum(model.a[k]*0.25 + model.b[k]*(0.5) for k in model.x_ind) + model.c  <= 0.5*(model.ysum - 2*model.L*model.xdist) 
  #---------------------------------------------------------------------
        
    @staticmethod
    def _quadratic_nonuniform(model, i):
        return model.f[i] == sum(model.a[j]*model.xs[i,j]**2 + model.b[j]*model.xs[i,j] for j in model.x_ind) + model.c 
  #---------------------------------------------------------------------
    
    @staticmethod
    def _exponential(model, i):
        a = sum(model.a[j]*(model.xs[i,j]-model.b[j])**2 for j in model.x_ind)
        return model.f[i] == pe.exp(a) + model.c
  #---------------------------------------------------------------------
    
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
    
  #---------------------------------------------------------------------
    
    
    @staticmethod
    def _make_pyomo_ddcu_nonuniform_with_LC(intercept):       
        ddcu = pe.AbstractModel()
        ddcu.sample_ind = pe.Set()
        ddcu.x_ind = pe.Set()  
        ddcu.endpoints = pe.Set()

        ddcu.ys = pe.Param(ddcu.sample_ind)
        ddcu.xs = pe.Param(ddcu.sample_ind, ddcu.x_ind)

        ddcu.a = pe.Var(ddcu.x_ind, within=pe.NonNegativeReals, initialize=0.)
        ddcu.b = pe.Var(ddcu.x_ind, within=pe.Reals)               
        ddcu.f = pe.Var(ddcu.sample_ind)
        ddcu.Rates = pe.Param(ddcu.sample_ind)
        if intercept:
            ddcu.c = pe.Var(within=pe.Reals)
        else:
            ddcu.c = pe.Param()    
        ddcu.delta = pe.Var(ddcu.endpoints, ddcu.x_ind, within=pe.Binary)

        ddcu.obj = pe.Objective(rule=DDCU_model._linear_obj_rule)
        ddcu.con1 = pe.Constraint(ddcu.sample_ind, rule=DDCU_model._underestimating_con_rule)
        ddcu.con2 = pe.Constraint(ddcu.sample_ind, rule=DDCU_model._quadratic_nonuniform)

        # Constraints for two diagonal points
        def _first_endpoint_rule(model, j):
            """
            Apply the negative Lipschitz rule for the interval lower bound.
            """
            i = min(model.endpoints)
            return DDCU_model._underestimating_lipschitz_neg_rule(model, i, j)

        def _last_endpoint_rule(model, j):
            """
            Apply the positive Lipschitz rule for the interval upper bound.
            """
            i = max(model.endpoints)
            return DDCU_model._underestimating_lipschitz_pos_rule(model, i, j)

        ddcu.con3_first_neg = pe.Constraint(ddcu.x_ind, rule=_first_endpoint_rule)
        ddcu.con3_last_pos = pe.Constraint(ddcu.x_ind, rule=_last_endpoint_rule)

        return ddcu
    
   #---------------------------------------------------------------------

    @staticmethod
    def _make_pyomo_ddcu_nonuniform_LC_with_bound(intercept):       
        ddcu = pe.AbstractModel()
        ddcu.sample_ind = pe.Set()
        ddcu.x_ind = pe.Set()  
        ddcu.endpoints = pe.Set()

        ddcu.ys = pe.Param(ddcu.sample_ind)
        ddcu.xs = pe.Param(ddcu.sample_ind, ddcu.x_ind)

        ddcu.a = pe.Var(ddcu.x_ind, within=pe.NonNegativeReals, initialize=0.)
        ddcu.b = pe.Var(ddcu.x_ind, within=pe.Reals)               
        ddcu.f = pe.Var(ddcu.sample_ind)
        ddcu.L = pe.Param(mutable=True)
        ddcu.ysum = pe.Param(mutable=True)
        ddcu.xdist = pe.Param(mutable=True)

        if intercept:
            ddcu.c = pe.Var(within=pe.Reals)
        else:
            ddcu.c = pe.Param()    

        ddcu.obj = pe.Objective(rule=DDCU_model._linear_obj_rule)
        ddcu.con1 = pe.Constraint(ddcu.sample_ind, rule=DDCU_model._underestimating_con_rule)
        ddcu.con2 = pe.Constraint(ddcu.sample_ind, rule=DDCU_model._quadratic_nonuniform)        
        ddcu.con3 = pe.Constraint(rule=DDCU_model._lipschitz_lb_on_underestimator_min_rule)

        return ddcu
    
   #---------------------------------------------------------------------
    
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
   
