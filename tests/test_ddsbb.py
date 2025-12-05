# tests/test_ddsbb.py

import time
import numpy as np
import pytest
import pyomo.environ as pe

from PyDDSBB.DDSBB import DDSBB
from PyDDSBB.DDSBBModel import Problem

INFINITY = np.inf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glpk_available():
    solver = pe.SolverFactory("glpk")
    return solver is not None and solver.available(False)


class DummySimulator:
    def __init__(self, sample_number=0, sense="minimize"):
        self.sample_number = sample_number
        self._sense = sense


class DummyBuilder:
    def __init__(self, sample_number=0, sense="minimize"):
        self.simulator = DummySimulator(sample_number=sample_number, sense=sense)
        self.node = 0


def make_simple_box_problem():
    """
    1D problem: f(x) = x^2 on [0,1], minimize.
    """
    def obj(x):
        return x[0] ** 2

    p = Problem()
    p.add_variable(0.0, 1.0, vartype="continuous")
    p.add_objective(obj, sense="minimize")
    return p


# ---------------------------------------------------------------------------
# DDSBB initialization and basic configuration
# ---------------------------------------------------------------------------

def test_ddsbb_init_defaults_and_sense_strings():
    d = DDSBB(number_init_samples=5)

    # Basic attributes
    assert d.init_sample == 5
    assert d.multifidelity is False
    assert d.split_method == "equal_bisection"
    assert d.variable_selection == "longest_side"
    assert d.underestimator_option == "Quadratic"

    # stop options should become attributes
    assert hasattr(d, "absolute_tolerance")
    assert hasattr(d, "relative_tolerance")
    assert hasattr(d, "sampling_limit")
    assert hasattr(d, "time_limit")

    # Minimize sense: LB/UB labels
    assert d.report_LB == "lower bound"
    assert d.report_UB == "upper bound"

    # Maximize sense flips LB / UB labels
    d2 = DDSBB(number_init_samples=3, sense="maximize")
    assert d2.report_LB == "upper bound"
    assert d2.report_UB == "lower bound"


def test_update_stop_criteria_sets_new_values_and_resets_stop():
    d = DDSBB(number_init_samples=3)
    d.stop = 3  # some nonzero state

    new_stop = {
        "absolute_tolerance": 1e-3,
        "sampling_limit": 123,
    }
    d.update_stop_criteria(new_stop)

    assert d.absolute_tolerance == 1e-3
    assert d.sampling_limit == 123
    # stop should be reset to 0
    assert d.stop == 0


# ---------------------------------------------------------------------------
# Adaptive sampling rule
# ---------------------------------------------------------------------------

def test_adaptive_static_and_instance_rule():
    dim = 2
    level = 1

    # static method
    expected = max(int(dim * 11 / level + 3), int(dim * 3) + 3)
    assert DDSBB._adaptive(dim, level) == expected

    # instance method _adaptive_rule
    d = DDSBB(number_init_samples=3)
    d.dim = dim
    d.level = level
    val = d._adaptive_rule()
    assert val == expected


# ---------------------------------------------------------------------------
# get_optimum / get_optimizer
# ---------------------------------------------------------------------------

def test_get_optimum_and_get_optimizer_minimize_and_maximize():
    # Minimize case: returns yopt_global as-is
    d_min = DDSBB(number_init_samples=3)
    d_min.builder = DummyBuilder(sample_number=0, sense="minimize")
    d_min.yopt_global = 1.23
    d_min.xopt_global = np.array([0.5])

    assert d_min.get_optimum() == pytest.approx(1.23)
    assert np.allclose(d_min.get_optimizer(), np.array([0.5]))

    # Maximize case: get_optimum returns -yopt_global
    d_max = DDSBB(number_init_samples=3)
    d_max.builder = DummyBuilder(sample_number=0, sense="maximize")
    d_max.yopt_global = -2.0
    d_max.xopt_global = np.array([0.25])

    assert d_max.get_optimum() == pytest.approx(2.0)
    assert np.allclose(d_max.get_optimizer(), np.array([0.25]))


# ---------------------------------------------------------------------------
# _check_resources (sampling limit and time limit)
# ---------------------------------------------------------------------------

def test_check_resources_reaches_sampling_limit():
    d = DDSBB(number_init_samples=3)
    d.builder = DummyBuilder(sample_number=10)  
    d.search_instance = 1
    d.time_start = time.time()
    d.time_limit = 1e6  
    d.sampling_limit = 5   
    d._completion_indicator = True

    d._check_resources()
    assert d.stop == 4
    assert d.stop_message == "reached sampling limit"


def test_check_resources_reaches_time_limit():
    d = DDSBB(number_init_samples=3)
    d.builder = DummyBuilder(sample_number=0)
    d.search_instance = 1
    d.time_start = time.time() - 10.0
    d.time_limit = 0.1
    d.sampling_limit = 100  
    d._completion_indicator = True

    d._check_resources()
    assert d.stop == 5
    assert d.stop_message == "reached time limit"


# ---------------------------------------------------------------------------
# _check_convergence branches
# ---------------------------------------------------------------------------

def _prepare_for_check_convergence(d):
    d._lowerbound_hist = []
    d._upperbound_hist = []
    d._sampling_hist = []
    d._lipschitz_hist = []
    d._cpu_hist = []
    d.builder = DummyBuilder(sample_number=0)
    d.search_instance = 1
    d.time_start = time.time()


def test_check_convergence_absolute_gap_closed():
    d = DDSBB(number_init_samples=3)
    _prepare_for_check_convergence(d)
    d.absolute_tolerance = 0.05
    d.relative_tolerance = 0.5
    d.min_xrange = 0.5
    d.lipschitz_current = 1.0
    d.flb_current = 0.96
    d.yopt_global = 1.0

    d._check_convergence()
    assert d.stop == 1
    assert d.stop_message == "absolute gap closed"
    assert d.lowerbound_global == 0.96


def test_check_convergence_relative_gap_closed():
    d = DDSBB(number_init_samples=3)
    _prepare_for_check_convergence(d)
    d.absolute_tolerance = 0.01
    d.relative_tolerance = 0.3
    d.min_xrange = 1.0
    d.lipschitz_current = 1.0
    d.flb_current = 0.8
    d.yopt_global = 1.0  

    d._check_convergence()
    assert d.stop == 2
    assert d.stop_message == "relative gap closed"


def test_check_convergence_min_xrange_too_small():
    d = DDSBB(number_init_samples=3)
    _prepare_for_check_convergence(d)
    d.absolute_tolerance = 0.01
    d.relative_tolerance = 0.1
    d.min_xrange = 0.01
    d.minimum_bound = 0.05
    d.lipschitz_current = 1.0
    d.flb_current = 0.0
    d.yopt_global = 1.0  

    d._check_convergence()
    assert d.stop == 3
    assert d.stop_message == "search space too small"


def test_check_convergence_infeasible_problem():
    d = DDSBB(number_init_samples=3)
    _prepare_for_check_convergence(d)
    d.flb_current = INFINITY
    d.lipschitz_current = INFINITY
    d.yopt_global = INFINITY

    d._check_convergence()
    assert d.stop == 4
    assert d.stop_message == "Problem Infeasible"


# ---------------------------------------------------------------------------
# Integration test: small box-constrained problem (GLPK only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not glpk_available(), reason="GLPK solver not available")
def test_ddsbb_optimize_simple_quadratic_box_problem():
    """
    End-to-end check: solve a small 1D box-constrained quadratic.
    This uses the Quadratic underestimator (GLPK-based) and no multifidelity.
    """
    problem = make_simple_box_problem()

    stop_option = {
        "absolute_tolerance": 1e-3,
        "relative_tolerance": 0.05,
        "minimum_bound": 0.01,
        "sampling_limit": 200,
        "time_limit": 60.0,
    }
    infeasible_limit = {"sampling_limit": 100, "time_limit": 10.0}

    solver = DDSBB(
        number_init_samples=5,
        multifidelity=False,
        split_method="equal_bisection",
        variable_selection="longest_side",
        underestimator_option="Quadratic",
        stop_option=stop_option,
        infeasible_limit=infeasible_limit,
        sense="minimize",
    )

    solver.optimize(problem)

    # Should have found a finite optimum
    opt = solver.get_optimum()
    xopt = solver.get_optimizer()

    assert np.isfinite(opt)
    assert xopt is not None
    # For f(x)=x^2 on [0,1], optimum is at x=0 with value 0
    assert opt >= -1e-6  # no negative
    assert opt <= 0.1    # near zero
    assert 0.0 <= xopt[0, 0] <= 1.0

    # sample number should be >= initial samples
    assert solver.builder.simulator.sample_number >= solver.init_sample
