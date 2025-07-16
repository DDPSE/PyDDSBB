#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on Tue Jul  7 13:32:23 2020
@Updated on Tue Jul 08 01:15:48 2025
@authors: JianyuanZhai, Suryateja Ravutla
"""

import time
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

class MachineLearning:
    """
    Base class for all machine learning model
    """
    def __init__(self, model = 'SVR'):
        self._model_type = 'SVR'
    def _predict(self, X, y, multifidelity):
        return self.model.predict(X)

class LocalSVR(MachineLearning):
    def __init__(self, eps = 0.01, var_select = True):
        """
        Inputs
        ------
        eps: float
             marginal width default 0.01
        var_select: bool
             decision to do variable selection 
        """
        super().__init__('SVR')
        self.eps = eps
        self.time_training = 0.
        self.time_var_select = 0.
    def _train(self, X, Y):
        """
        Inputs
        ------
        X: ndarray of shape(n_samples, n_variables)
           input data
        Y: ndarray of shape(n_samples, 1)
           output data
        """
        time_start = time.time()
        self.dim = X.shape[1]
        y_avg = np.average(Y)
        dev = 3*np.std(Y)
        cost = max(abs(y_avg+dev), abs(y_avg-dev))
        self._gamma = 1/(self.dim * X.var())
        svr_rbf = SVR(kernel = 'rbf', C = cost, epsilon = self.eps, gamma = self._gamma)
        self.model = svr_rbf.fit(X, Y)  
        self.time_training += time.time() - time_start
    def _rank(self):
        """
        Calculate SVR-based criterion value
        """
        time_start = time.time()
        sv = self.model.support_vectors_
        coef = self.model.dual_coef_[0]
        norm2 = self._gamma*np.square(sv - sv[:, None])
        norm = np.exp(-np.sum(norm2, axis = -1))
        coef_product = coef[:,None]*coef
        crit = np.sum(np.sum(norm*coef_product*norm2.T, 1),1)       
        self.time_var_select += time.time() - time_start
        return crit