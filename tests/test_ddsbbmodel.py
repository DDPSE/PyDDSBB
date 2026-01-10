# tests/test_ddsbbmodel.py

import numpy as np
import pytest

from PyDDSBB.DDSBBModel import Var, Problem


# -----------------------------------------------------------------------------
# Var tests
# -----------------------------------------------------------------------------

def test_var_initialization_and_getters():
    lb, ub = -1.5, 2.5
    vartype = "continuous"

    v = Var(lb, ub, vartype)

    # bounds stored as numpy array [lb, ub]
    bounds = v._get_bound()
    assert isinstance(bounds, np.ndarray)
    assert bounds.shape == (2,)
    assert np.allclose(bounds, np.array([lb, ub]))

    # vartype getter
    assert v._get_vartype() == vartype


def test_var_can_handle_integer_type():
    v = Var(0, 10, "integer")
    assert np.allclose(v._get_bound(), np.array([0, 10]))
    assert v._get_vartype() == "integer"


# -----------------------------------------------------------------------------
# Problem initialization
# -----------------------------------------------------------------------------

def test_problem_initial_state():
    p = Problem()

    assert p._dim == 0
    assert p._number_known_constraint == 0
    assert p._number_unknown_constraint == 0
    assert isinstance(p._variable, list)
    assert isinstance(p._type, list)
    assert p._variable == []
    assert p._type == []


# -----------------------------------------------------------------------------
# add_variable behavior
# -----------------------------------------------------------------------------

def test_add_single_variable_updates_dim_and_type():
    p = Problem()
    p.add_variable(lb=0.0, ub=1.0, vartype="continuous")

    assert p._dim == 1
    assert len(p._variable) == 1
    assert isinstance(p._variable[0], Var)

    # vartype stored in _type
    assert p._type == ["continuous"]

    # bounds from stored Var
    b0 = p._variable[0]._get_bound()
    assert np.allclose(b0, np.array([0.0, 1.0]))


def test_add_multiple_variables_and_types():
    p = Problem()

    p.add_variable(0.0, 1.0, "continuous")
    p.add_variable(-1, 5, "integer")
    p.add_variable(2, 10, "integer")   # duplicate type

    assert p._dim == 3
    assert len(p._variable) == 3

    # _type should have unique vartypes in order of first appearance
    assert "continuous" in p._type
    assert "integer" in p._type
    assert len(p._type) == 2

    # check bounds of second and third
    b1 = p._variable[1]._get_bound()
    b2 = p._variable[2]._get_bound()
    assert np.allclose(b1, np.array([-1, 5]))
    assert np.allclose(b2, np.array([2, 10]))


# -----------------------------------------------------------------------------
# add_objective and update_sense
# -----------------------------------------------------------------------------

def test_add_objective_sets_objective_and_default_sense():
    p = Problem()

    def obj(x):
        return x[0] + 2.0

    p.add_objective(obj)   # default sense = 'minimize'

    assert p._objective is obj
    assert p._sense == "minimize"


def test_add_objective_with_explicit_sense_and_update_sense():
    p = Problem()

    def obj(x):
        return -x[0]

    p.add_objective(obj, sense="maximize")
    assert p._objective is obj
    assert p._sense == "maximize"

    # change sense later
    p.update_sense("minimize")
    assert p._sense == "minimize"


# -----------------------------------------------------------------------------
# known and unknown constraints
# -----------------------------------------------------------------------------

def test_add_known_constraint_increments_counter_and_sets_attribute():
    p = Problem()

    c1 = "x[0] + x[1] <= 1.0"
    c2 = "x[0] - x[1] >= 0.0"

    p.add_known_constraint(c1)
    assert p._number_known_constraint == 1
    # attribute _known_constraint1 should exist and equal c1
    assert hasattr(p, "_known_constraint1")
    assert getattr(p, "_known_constraint1") == c1

    p.add_known_constraint(c2)
    assert p._number_known_constraint == 2
    assert hasattr(p, "_known_constraint2")
    assert getattr(p, "_known_constraint2") == c2


def test_add_unknown_constraint_increments_counter_and_sets_attribute():
    p = Problem()

    def uc1(x):
        return 1 if x[0] <= 0.5 else 0

    def uc2(x):
        return 1 if x[1] >= 0.0 else 0

    p.add_unknown_constraint(uc1)
    assert p._number_unknown_constraint == 1
    assert hasattr(p, "_unknown_constraint1")
    assert getattr(p, "_unknown_constraint1") is uc1

    p.add_unknown_constraint(uc2)
    assert p._number_unknown_constraint == 2
    assert hasattr(p, "_unknown_constraint2")
    assert getattr(p, "_unknown_constraint2") is uc2
