#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:44:11 2021

@author: JianyuanZhai
"""

from setuptools import setup

setup(
      name = 'PyDDSBB',
      version = '0.1.3',
      description = 'Data-driven Spatial Branch-and-bound Algorithm for Python',
      author = ['Jianyuan Zhai', 'Fani Boukouvala'],
      author_email = ['zhaijianyuan@gmail.com', 'fani.boukouvala@chbe.gatech.edu'],
      license = 'MIT',
      packages = ['PyDDSBB'],
      install_requires = ['numpy','pyomo','scikit-learn']
      )
