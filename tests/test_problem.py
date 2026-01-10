# tests/test_problem.py

import numpy as np
import pytest

from PyDDSBB.DDSBBModel import Problem
from PyDDSBB._problem import BoundConstrainedSimulation, BlackBoxSimulation, GreyBoxSimulation


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_1d_quadratic_problem(sense="minimize"):
    """
    1D problem: f(x) = x^2 on [0,1].
    """
    def obj(x):
        # x is a 1D numpy array
        return x[0] ** 2

    prob = Problem()
    prob.add_variable(0.0, 1.0, vartype="continuous")
    prob.add_objective(obj, sense=sense)
    return prob


def make_1d_blackbox_problem():
    """
    1D problem: objective f(x) = x, unknown constraint: x <= 0.5.
    Unknown constraints must return 1 if satisfied, 0 otherwise.
    """
    def obj(x):
        return x[0]

    def uc1(x):
        return 1 if x[0] <= 0.5 else 0

    prob = Problem()
    prob.add_variable(0.0, 1.0, vartype="continuous")
    prob.add_objective(obj, sense="minimize")
    prob.add_unknown_constraint(uc1)
    return prob, uc1


def make_greybox_problem():
    """
    1D problem with one known constraint (string) and one unknown constraint (function).
    """
    def obj(x):
        return x[0] ** 2

    def uc1(x):
        return 1 if x[0] <= 0.7 else 0

    known_str = "x[0] - 0.5 <= 0"

    prob = Problem()
    prob.add_variable(0.0, 1.0, vartype="continuous")
    prob.add_objective(obj, sense="minimize")
    prob.add_known_constraint(known_str)
    prob.add_unknown_constraint(uc1)
    return prob, known_str, uc1


# -----------------------------------------------------------------------------
# BoundConstrainedSimulation tests
# -----------------------------------------------------------------------------

def test_boundconstrained_simulation_minimize_basic():
    prob = make_1d_quadratic_problem(sense="minimize")
    sim = BoundConstrainedSimulation(prob)

    # Class-level type
    assert sim.problem_type == "BoundConstrained"

    # Check internal attributes
    assert sim._dim == 1
    assert sim._bounds.shape == (2, 1)
    assert sim._sense == "minimize"
    assert sim._objective is prob._objective

    # _simulate should be _obj_minimize for "minimize"
    assert sim._simulate.__func__ is BoundConstrainedSimulation._obj_minimize

    # time_sampling & sample_number start at 0
    assert sim.time_sampling == 0.0
    assert sim.sample_number == 0

    # Evaluate at multiple points
    X = np.array([[0.0],
                  [0.5],
                  [1.0]])
    y = sim._obj_minimize(X)
    assert isinstance(y, np.ndarray)
    assert y.shape == (3,)
    # y = x^2
    assert np.allclose(y, np.array([0.0, 0.25, 1.0]))

    # sample_number updated
    assert sim.sample_number == 3

    # Single point should return float
    X1 = np.array([[0.3]])
    y1 = sim._obj_minimize(X1)
    assert isinstance(y1, float)
    assert y1 == pytest.approx(0.3 ** 2)
    assert sim.sample_number == 4  # +1


def test_boundconstrained_simulation_maximize_basic():
    prob = make_1d_quadratic_problem(sense="maximize")
    sim = BoundConstrainedSimulation(prob)

    assert sim.problem_type == "BoundConstrained"
    assert sim._sense == "maximize"
    # _simulate should be _obj_maximize for "maximize"
    assert sim._simulate.__func__ is BoundConstrainedSimulation._obj_maximize

    X = np.array([[0.0],
                  [0.5],
                  [1.0]])
    y = sim._obj_maximize(X)

    # maximizing x^2 is implemented as minimizing -x^2
    assert np.allclose(y, -np.array([0.0, 0.25, 1.0]))
    assert sim.sample_number == 3

    # Single point returns float
    X1 = np.array([[0.3]])
    y1 = sim._obj_maximize(X1)
    assert isinstance(y1, float)
    assert y1 == pytest.approx(-(0.3 ** 2))
    assert sim.sample_number == 4


# -----------------------------------------------------------------------------
# BlackBoxSimulation tests
# -----------------------------------------------------------------------------

def test_blackbox_simulation_initialization_and_unknown_constraints():
    prob, uc1 = make_1d_blackbox_problem()
    bb = BlackBoxSimulation(prob)

    assert bb.problem_type == "BlackBox"
    # Inherits dimension and bounds from BoundConstrainedSimulation
    assert bb._dim == 1
    assert bb._bounds.shape == (2, 1)

    # Unknown constraints correctly set
    assert bb._number_unknown_constraint == 1
    assert len(bb._unknown_constraints) == 1
    assert bb._unknown_constraints[0] is uc1

    # time_constraints initialized to 0
    assert bb.time_constraints == 0


def test_blackbox_check_feasibility_mixed():
    prob, uc1 = make_1d_blackbox_problem()
    bb = BlackBoxSimulation(prob)

    X = np.array([[0.2],
                  [0.7],
                  [0.5],
                  [1.0]])

    feas = bb._check_feasibility(X)

    # 0.2, 0.5 satisfy x <= 0.5; others do not
    expected = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.allclose(feas, expected)

    # BoundConstrainedSimulation.sample_number is *not* updated by _check_feasibility
    assert bb.sample_number == 0


def test_blackbox_check_feasibility_all_feasible_or_all_infeasible():
    prob, uc1 = make_1d_blackbox_problem()
    bb = BlackBoxSimulation(prob)

    X_all_feas = np.array([[0.1],
                           [0.3],
                           [0.5]])
    feas_all = bb._check_feasibility(X_all_feas)
    assert np.all(feas_all == 1.0)

    X_all_infeas = np.array([[0.6],
                             [0.8]])
    feas_none = bb._check_feasibility(X_all_infeas)
    assert np.all(feas_none == 0.0)


# -----------------------------------------------------------------------------
# GreyBoxSimulation tests
# -----------------------------------------------------------------------------

def test_greybox_simulation_initialization_known_and_unknown():
    prob, known_str, uc1 = make_greybox_problem()
    gb = GreyBoxSimulation(prob)

    assert gb.problem_type == "GreyBox"

    assert gb._number_unknown_constraint == 1
    assert gb._number_known_constraint == 1

    # unknown constraints: same callable as in the Problem
    assert len(gb._unknown_constraints) == 1
    assert gb._unknown_constraints[0] is uc1

    # known constraints: strings with " + eps1" suffix
    assert len(gb._known_constraints) == 1
    assert isinstance(gb._known_constraints[0], str)
    assert gb._known_constraints[0] == known_str + " + eps1"

    assert gb.time_constraints == 0


def test_greybox_inherits_boundconstrained_behavior():
    prob, known_str, uc1 = make_greybox_problem()
    gb = GreyBoxSimulation(prob)

    # Should still behave like a BoundConstrainedSimulation for objectives
    X = np.array([[0.0],
                  [0.5],
                  [1.0]])

    y = gb._simulate(X)  # based on sense='minimize' in make_greybox_problem
    assert y.shape == (3,)
    assert np.allclose(y, np.array([0.0, 0.25, 1.0]))
    assert gb.sample_number == 3
