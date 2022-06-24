---
title: 'PyDDSBB: A Python Package for Simulation-Optimization using Data-Driven Branch-and-Bound Techniques'
tags:
  - Python
  - Data-Driven Optimization
  - Simulation Optimization
  - Multi-fidelity
  - Branch-and-Bound
authors:
  - name: Jianyuan Zhai
    orcid: 0000-0001-7544-793X
    affiliation: 1
  - name: Suryateja Ravutla
    orcid: 0000-0002-5039-8032
    affiliation: 1
  - name: Fani Boukouvala
    orcid: 0000-0002-0584-1517
    corresponding: true
    affiliation: 1
affiliations:
 - name: Department of Chemical and Biomolecular Engineering, Georgia Institute of Technology, Atlanta, GA
   index: 1
date: 20 June 2022
bibliography: paper.bib

---

# Summary

High-fidelity (HF) computer simulations are essential for quantitative analysis and decision-making (i.e., design or inverse optimization), however, often the objective functions and the constraints reliant on a simulation are only available as black-box functions [@McBride2019OverviewEngineering; @Rios2013Derivative-freeImplementations; @Boukouvala2016GlobalCDFO; @Bhosekar2018AdvancesReview]. Sample generation using HF simulations is often expensive due to computational cost and time. Most of the equation/derivative-based optimization solvers cannot be directly applied to optimize problems with embedded HF simulations due to lack of closed-form equations and gradient information. As an alternative, machine learning (ML) surrogate models are often employed to approximate HF simulation data and expedite the optimization search[@Bhosekar2018AdvancesReview]. However, training and building accurate models requires large amount of data model parameterizations are highly subjected to the available data.  One may arrive at different solutions because the ML models may be nonconvex and the surrogate model parameterizations can be different due to re-initialization and slight changes in the data set. Finally, most derivative-free or black-box methods offer no information regarding the quality of the incumbent optimal solution (e.g., upper/lower bound on the optimum) [@Boukouvala2016GlobalCDFO; @Amaran2016SimulationApplications; @Larson2019Derivative-freeMethods; @Zhai2021Data-drivenOptimization; @Zhai2022Data-drivenOptimization]. To tackle these challenges, we recently proposed a data-driven equivalent of the spatial branch-and-bound (DDSBB) algorithm based on the concept of constructing convex underestimators of HF data from simulations and low-fidelity (LF) data from ML model approximations [@Zhai2021Data-drivenOptimization; @Zhai2022Data-drivenOptimization].

# Statement of need

Black-box optimization is a challenging problem due to lack of analytic equations and derivatives. Such problems arise in many fields of science and engineering, such as chemical process design, oilfield operations, protein folding, aircract/vehicle design, and many more [@McBride2019OverviewEngineering]. Developing novel and improved data-driven approaches for efficient optimization of such systems will help in quantitative analysis and improved decision making. Although many methods exist, one drawback of many sampling- or ML-based optimization methods is the high dependency on initial sampling and selection of the ML model type. PyDDSBB is a Python package for the proposed Data-Driven Spatial Branch-and-Bound Algorithm[@Zhai2021Data-drivenOptimization; @Zhai2022Data-drivenOptimization]. Instead of only using samples, or directly relying on a single ML model fit, pyDDSBB develops convex underestimators as relaxations of data generated from HF models and/or ML surrogates. These relaxations are embedded into the branch and bound algorithm, and the search space is progressively partitioned by branching and pruning heuristics. This approach allows for estimation of upper and lower bounds on the optimum solution, which help in pruning spaces that have a very low probability of containing a better optimum. Samples are added adaptively in the non-pruned subspaces[@Rios2013Derivative-freeImplementations; @Zhai2021Data-drivenOptimization; @Zhai2022Data-drivenOptimization]. Through benchmarking of pyDDSBB on a large pool of problems with dimensions 1-10, we have shown that the solver can locate the global optimum for >90% of the problems with 2-3 dimensions and >70% of the problems with 4-10 dimensions. Most importantly pyDDSBB provides valid bounds for overall >90% of the problems[@Zhai2022Data-drivenOptimization].

# Overview and Description

The PyDDSBB algorithm has been constructed using object-oriented programming, with an intent of easier incorporation of further additions or extensions to the algorithm. The algorithm has the option of employing Support Vector Regression models (trained by HF data) to generate LF data that also inform the underestimators. PyDDSBB utilizes the Scikit-learn [@Pedregosa2011Scikit-learn:Python] package for constructing and optimizing the surrogate models for LF data generation and also offers the capability of user-based additions of new surrogate models. The algorithm utilizes the package – PYOMO[@Hart2011Pyomo:Python] to construct the convex quadratic underestimators and also provides an option for integration of other type of underestimators. Latin Hypercube Sampling (LHS) technique is used for generating the samples and the samples are generated adaptively in the non-pruned subspaces by utilizing the bounding information. Two possible paths for sampling and constructing the underestimators in algorithm are shown in \autoref{fig:MF}.

![Overview of PyDDSBB sampling and understimator construction. Solid red line shows the HF path where only HF samples obtained from black-box simulation are used in constructing underestimators. Dotted red line shows the path for MF approach, in which HF samples are used to generete LF samples and, combined HF and LF samples are used to construct underestimators \label{fig:MF}](MF.jpg)

While the objective function is assumed to be simulation-based (black-box), PyDDSBB allows the user to include simulation-based constraints (unknown or black-box) and equation-based (known) constraints into the formulation. A variety of branching techniques (e.g., equal bisection, longest side, ML-based prioritizing important variables, etc.)[@Zhai2021Data-drivenOptimization] and branch-and-bound heuristics are considered, and default options are recommended for the case of box-constrained and general constrained problems. The HF simulation can be in the form of a python function, or any external platform that could be connected via an API. The ultimate goal of this software is to provide a user-friendly simulation-based optimization framework for both expert and non-expert users, benefiting from the high-level features of Python. PyDDSBB follows the object-oriented programming paradigm and is designed to allow easy extension of the core functionality by users and developers. The optimization of an example benchmark black-box function and visualization of the outputs in shown in \autoref{fig:rO}

![Optimization of sample benchmark function and visualizing the output. **Top left:** A non-convex benchmark function $f(x_1, x_2) = 4x_1^2 - 2.1x_1^4 + 0.3333x_1^6 + x_1x_2 - 4x_2^2 + 4x_2^4$, available as black-box model. **Top right:** Evolution of upper and lower bounds found during the branching process of the sample space. **Bottom right:** Sampling and branching in the sample space visualized for the branch-and-bound process. Darker regions correspond to a higher level of branching and the red dot represents the optimum solution found. **Bottom left:** Algorithm output after termination with some information on sampling and optimum solution found \label{fig:rO}](resultsOverview.jpg)

# Acknowledgements

The authors acknowledge financial support from the National Science Foundation (NSF-1805724 & GR00003313) and DOE/AICHE RAPID Institute "Synopsis" project grant (GR10002225)

# References
