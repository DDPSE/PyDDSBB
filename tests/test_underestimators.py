# tests/test_underestimtators.py

import numpy as np
import pytest
import pyomo.environ as pe

from PyDDSBB._underestimators import (
    DDCU_Nonuniform,
    DDCU_Nonuniform_with_LC,
    DDCU_Nonuniform_with_LC_and_IC,
    DDCU_Nonuniform_with_LC_and_bound,
    DDCU_Nonuniform_with_LC_and_IC_and_bound,
)


# =============================================================================
# Helpers
# =============================================================================

def make_quadratic_data_1d(n=5):
    """Simple f(x) = x^2 on [0,1]."""
    x = np.linspace(0.0, 1.0, n)[:, None]
    y = x[:, 0] ** 2
    return x, y


def make_simple_2d_data():
    """
    Simple 2D dataset, roughly linear: f(x) = x1 + 2*x2.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )
    Y = X[:, 0] + 2.0 * X[:, 1]
    return X, Y


# =============================================================================
# _minimize_1d_vec tests 
# =============================================================================

def test_minimize_1d_vec_basic_cases():
    """
    Check the branching logic in _minimize_1d_vec:
    - a > 0, interior optimum in [0,1]
    - a > 0, optimum < 0  -> x = 0
    - a > 0, optimum > 1  -> x = 1
    - a = 0, b > 0 -> x = 0
    - a = 0, b < 0 -> x = 1
    - a = 0, b ≈ 0 -> x = 0.5
    """
    cls = DDCU_Nonuniform  # all classes share the same implementation

    # Cases:
    # 0: a>0, b=-1 -> optimum at 0.5
    # 1: a>0, b= 4 -> optimum -b/(2a) = -2 < 0 -> x = 0
    # 2: a>0, b=-4 -> optimum 2 > 1 -> x = 1
    # 3: a=0, b= 1 -> x = 0
    # 4: a=0, b=-1 -> x = 1
    # 5: a=0, b≈0 -> x = 0.5
    a = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    b = np.array([-1.0, 4.0, -4.0, 1.0, -1.0, 0.0])
    c = 0.0

    xopt = cls._minimize_1d_vec(a, b, c)

    assert np.isclose(xopt[0], 0.5)   # interior in [0,1]
    assert np.isclose(xopt[1], 0.0)   # optimum < 0 -> 0
    assert np.isclose(xopt[2], 1.0)   # optimum > 1 -> 1
    assert np.isclose(xopt[3], 0.0)   # a=0, b>0 -> 0
    assert np.isclose(xopt[4], 1.0)   # a=0, b<0 -> 1
    assert np.isclose(xopt[5], 0.5)   # a=0, b≈0 -> 0.5


# =============================================================================
# estimate_lipschitz_constant tests
# =============================================================================

@pytest.mark.parametrize(
    "cls",
    [
        DDCU_Nonuniform_with_LC,
        DDCU_Nonuniform_with_LC_and_IC,
        DDCU_Nonuniform_with_LC_and_bound,
        DDCU_Nonuniform_with_LC_and_IC_and_bound,
    ],
)
def test_estimate_lipschitz_constant_empty_raises(cls):
    X = np.zeros((0, 1))
    Y = np.zeros((0,))
    with pytest.raises(ValueError):
        cls.estimate_lipschitz_constant(X, Y)


@pytest.mark.parametrize(
    "cls",
    [
        DDCU_Nonuniform_with_LC,
        DDCU_Nonuniform_with_LC_and_IC,
        DDCU_Nonuniform_with_LC_and_bound,
        DDCU_Nonuniform_with_LC_and_IC_and_bound,
    ],
)
def test_estimate_lipschitz_constant_single_sample_zero(cls):
    X = np.array([[0.0]])
    Y = np.array([1.0])
    maxL, rates = cls.estimate_lipschitz_constant(X, Y, n_neighbors=3)

    assert maxL == 0.0
    assert rates == {0: 0.0}


@pytest.mark.parametrize(
    "cls",
    [
        DDCU_Nonuniform_with_LC,
        DDCU_Nonuniform_with_LC_and_IC,
        DDCU_Nonuniform_with_LC_and_bound,
        DDCU_Nonuniform_with_LC_and_IC_and_bound,
    ],
)
def test_estimate_lipschitz_constant_basic_neighbors(cls):
    """
    Basic sanity check with more than one point and n_neighbors > n-1
    to trigger the internal adjustment + warning.
    """
    X = np.array([[0.0], [1.0], [2.0]])
    Y = np.array([0.0, 1.0, 4.0])

    with pytest.warns(UserWarning):
        maxL, rates = cls.estimate_lipschitz_constant(X, Y, n_neighbors=10)

    assert maxL >= 0.0
    assert len(rates) == X.shape[0]
    assert all(np.isclose(v, maxL) for v in rates.values())


# =============================================================================
# GLPK-based underestimator tests
# =============================================================================

def test_ddcu_nonuniform_underestimate_quadratic_intercept_true():
    """
    DDCU_Nonuniform with intercept=True, simple 1D quadratic.
    Uses GLPK – no Gurobi dependency.
    """
    all_X, all_Y = make_quadratic_data_1d(n=5)
    ue = DDCU_Nonuniform(intercept=True)

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=None,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=None,
    )

    # flb_s should be a float and a valid lower bound (up to tolerance)
    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5

    # maxL is not used here; code returns np.nan
    assert np.isnan(maxL)

    assert isinstance(xopt, np.ndarray)
    assert xopt.shape == (1, all_X.shape[1])
    assert np.all(xopt >= 0.0) and np.all(xopt <= 1.0)
    assert ue.time_underestimate > 0.0


def test_ddcu_nonuniform_underestimate_quadratic_intercept_false():
    """
    DDCU_Nonuniform with intercept=False, ensuring it handles the corner point
    at the origin correctly.
    """
    all_X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    all_Y = all_X[:, 0] ** 2

    ue = DDCU_Nonuniform(intercept=False)

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=None,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=None,
    )

    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5
    assert np.isnan(maxL)
    assert xopt.shape == (1, 1)
    assert 0.0 <= xopt[0, 0] <= 1.0


def test_ddcu_nonuniform_with_LC_and_bound_underestimate_quadratic():
    """
    GLPK-based Lipschitz + bound model.
    """
    all_X, all_Y = make_quadratic_data_1d(n=5)
    ue = DDCU_Nonuniform_with_LC_and_bound(intercept=True)

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=None,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=None,
    )

    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5
    assert isinstance(maxL, float)
    assert maxL >= 0.0

    assert xopt.shape == (1, all_X.shape[1])
    assert np.all(xopt >= 0.0) and np.all(xopt <= 1.0)
    assert ue.time_underestimate > 0.0


def test_ddcu_nonuniform_with_LC_and_IC_and_bound_underestimate_IR_high():
    """
    GLPK-based LC+IC+bound model with IR >= 0.1 so the LC_with_bound model is used.
    """
    all_X, all_Y = make_simple_2d_data()
    ue = DDCU_Nonuniform_with_LC_and_IC_and_bound(intercept=True)

    xrange = np.array([1.0, 1.0])
    overallXrange = np.array([1.0, 1.0])  # IR = 1.0 >= 0.1

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=xrange,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=overallXrange,
    )

    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5

    assert isinstance(maxL, float)
    assert maxL >= 0.0

    assert xopt.shape == (1, all_X.shape[1])
    assert np.all(xopt >= 0.0) and np.all(xopt <= 1.0)
    assert ue.time_underestimate > 0.0


# =============================================================================
# Optional Gurobi-based underestimator tests
# =============================================================================

def _gurobi_available():
    solver = pe.SolverFactory("gurobi")
    return solver is not None and solver.available(False)


@pytest.mark.skipif(not _gurobi_available(), reason="Gurobi not available")
def test_ddcu_nonuniform_with_LC_underestimate_quadratic_gurobi():
    """
    Requires Gurobi. If not installed, this test is skipped (not failed).
    """
    all_X, all_Y = make_quadratic_data_1d(n=5)
    ue = DDCU_Nonuniform_with_LC(intercept=True)

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=None,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=None,
    )

    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5
    assert isinstance(maxL, float)
    assert maxL >= 0.0
    assert xopt.shape == (1, all_X.shape[1])
    assert np.all(xopt >= 0.0) and np.all(xopt <= 1.0)


@pytest.mark.skipif(not _gurobi_available(), reason="Gurobi not available")
def test_ddcu_nonuniform_with_LC_and_IC_underestimate_quadratic_gurobi():
    """
    Requires Gurobi. Here we also pass an IR that triggers the LC model,
    but the internal choice is not asserted – we just check a valid result.
    """
    all_X, all_Y = make_quadratic_data_1d(n=5)
    ue = DDCU_Nonuniform_with_LC_and_IC(intercept=True)

    xrange = np.array([1.0])
    overallXrange = np.array([1.0])  # IR = 1.0

    flb_s, maxL, xopt = ue._underestimate(
        all_X=all_X,
        all_Y=all_Y,
        lowfidelity_X=None,
        lowfidelity_Y=None,
        xrange=xrange,
        yrange=None,
        xbounds=None,
        ymin_local=None,
        overallXrange=overallXrange,
    )

    assert isinstance(flb_s, float)
    assert flb_s <= np.min(all_Y) + 1e-5
    assert isinstance(maxL, float)
    assert maxL >= 0.0
    assert xopt.shape == (1, all_X.shape[1])
    assert np.all(xopt >= 0.0) and np.all(xopt <= 1.0)
