# tests/test_tutorial_examples.py
import numpy as np
import PyDDSBB
import pytest


# ---------------------------------------------------------------------
# Helper functions (same examples as the tutorial notebook)
# ---------------------------------------------------------------------

def multi_gauss(x):
    """2D multi-modal Gaussian test function used in the tutorial."""
    x0, x1 = x
    f = (
        -0.5 * np.exp(-100.0 * (x0**2 + x1**2))
        - 1.2 * np.exp(-4.0 * ((x0 - 1.0)**2 + x1**2))
        - 1.0 * np.exp(-3.0 * (x0**2 + (x1 + 0.5)**2))
        - 1.0 * np.exp(-2.0 * ((x0 + 0.5)**2 + x1**2))
        - 1.2 * np.exp(-4.0 * (x0**2 + (x1 - 1.0)**2))
    )
    return float(f)


def prob06(x):
    """Same definition as in the tutorial: minimize x[0]."""
    return float(x[0])


def prob06_cons(x):
    """
    Simple black-box feasibility indicator used in README/test.py:
    intersection of two circles. 1.0 = feasible, 0.0 = infeasible.
    """
    x0, x1 = x
    c1 = (x0 - 2.0) ** 2 + (x1 - 4.0) ** 2 <= 4.0
    c2 = (x0 - 3.0) ** 2 + (x1 - 3.0) ** 2 <= 4.0
    return 1.0 if (c1 and c2) else 0.0


@pytest.fixture(scope="module")
def multi_gauss_model():
    model = PyDDSBB.DDSBBModel.Problem()
    model.add_objective(multi_gauss, sense="minimize")
    model.add_variable(-1.0, 1.0)
    model.add_variable(-1.0, 1.0)
    return model


@pytest.fixture(scope="module")
def quadratic_solver(multi_gauss_model):
    np.random.seed(100)
    solver = PyDDSBB.DDSBB(
        21,
        underestimator_option="Quadratic",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 200,
            "time_limit": 5000,
        },
    )
    solver.optimize(multi_gauss_model)
    return solver


@pytest.fixture(scope="module")
def lipschitz_qub_solver(multi_gauss_model):
    np.random.seed(100)
    solver = PyDDSBB.DDSBB(
        21,
        underestimator_option="Lipschitz-QUB",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 200,
            "time_limit": 5000,
        },
    )
    solver.optimize(multi_gauss_model)
    return solver


@pytest.fixture(scope="module")
def hybrid_lipschitz_qub_solver(multi_gauss_model):
    np.random.seed(100)
    solver = PyDDSBB.DDSBB(
        21,
        underestimator_option="Hybrid-Lipschitz-QUB",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 200,
            "time_limit": 5000,
        },
    )
    solver.optimize(multi_gauss_model)
    return solver


@pytest.fixture(scope="module")
def blk_model():
    """Black-box constrained model (prob06 + prob06_cons) as in the tutorial."""
    model = PyDDSBB.DDSBBModel.Problem()
    model.add_objective(prob06, sense="minimize")
    model.add_unknown_constraint(prob06_cons)
    model.add_variable(1.0, 5.5)
    model.add_variable(1.0, 5.5)
    return model


@pytest.fixture(scope="module")
def gr_model():
    """
    Grey-box version of prob06:
    includes both the unknown black-box constraint and one known constraint.
    """
    model = PyDDSBB.DDSBBModel.Problem()
    model.add_objective(prob06, sense="minimize")
    model.add_unknown_constraint(prob06_cons)
    model.add_known_constraint("(x0-2.0)**2 + (x1-4.0)**2 <= 4.0")
    model.add_variable(1.0, 5.5)
    model.add_variable(1.0, 5.5)
    return model


@pytest.fixture(scope="module")
def blk_solver(blk_model):
    np.random.seed(1)
    solver = PyDDSBB.DDSBB(
        23,
        underestimator_option="Quadratic",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 300,
            "time_limit": 5000,
        },
    )
    solver.optimize(blk_model)
    return solver


@pytest.fixture(scope="module")
def gr_solver(gr_model):
    np.random.seed(1)
    solver = PyDDSBB.DDSBB(
        23,
        underestimator_option="Quadratic",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 300,
            "time_limit": 5000,
        },
    )
    solver.optimize(gr_model)
    return solver


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_multi_gauss_model_dimension(multi_gauss_model):
    """Problem dimensionality in the tutorial should be 2."""
    assert multi_gauss_model._dim == 2


def test_quadratic_underestimator_returns_feasible_point(quadratic_solver):
    """Basic sanity check: solver returns a feasible optimizer in the variable bounds."""
    xopt = quadratic_solver.get_optimizer()
    yopt = quadratic_solver.get_optimum()

    xopt = np.asarray(xopt)

    assert isinstance(yopt, float)
    assert isinstance(xopt, np.ndarray)

    xopt = xopt.reshape(-1)
    assert xopt.size == 2

    assert np.all(xopt >= -1.0 - 1e-8)
    assert np.all(xopt <= 1.0 + 1e-8)

    # multi_gauss has y_opt < 0,
    # so DDSBB must find a value < 0.
    assert yopt < 0.0



def test_resume_does_not_worsen_solution(multi_gauss_model):
    """
    Mirror the tutorial's .resume() call:
    - run with a small sampling_limit,
    - then resume with a higher sampling_limit and check that
      the solution does not become worse.
    """
    np.random.seed(123)

    solver = PyDDSBB.DDSBB(
        21,
        underestimator_option="Quadratic",
        split_method="equal_bisection",
        variable_selection="longest_side",
        multifidelity=False,
        stop_option={
            "absolute_tolerance": 0.05,
            "relative_tolerance": 0.01,
            "minimum_bound": 0.05,
            "sampling_limit": 80,
            "time_limit": 5000,
        },
    )
    solver.optimize(multi_gauss_model)
    y_initial = solver.get_optimum()

    solver.resume({"sampling_limit": 200})# Resume with a larger sampling_limit
    y_resumed = solver.get_optimum()

    # Minimization: resumed solution should not be worse
    assert y_resumed <= y_initial + 1e-8


@pytest.mark.parametrize(
    "fixture_name",
    ["quadratic_solver", "lipschitz_qub_solver", "hybrid_lipschitz_qub_solver"],
)
def test_different_underestimators_give_similar_values(request, fixture_name):
    """
    Use the three underestimator options from the tutorial and
    check they all find reasonably similar objective values on multi_gauss.
    """
    solver = request.getfixturevalue(fixture_name)
    y = solver.get_optimum()
    assert np.isfinite(y)

    # All of them should be "close" to each other *** for this case only!:
    quad_solver = request.getfixturevalue("quadratic_solver")
    y_quad = quad_solver.get_optimum()
    assert abs(y - y_quad) < 0.5


def test_black_box_and_grey_box_solutions_are_feasible_and_consistent(
    blk_solver, gr_solver
):
    """
    From prob06 example:
    - black-box model uses only prob06_cons as unknown constraint
    - grey-box model adds one known constraint string.
    We expect:
      * both optimizers to be feasible for prob06_cons
      * their objective values to be close.
    """
    x_blk = blk_solver.get_optimizer()
    y_blk = blk_solver.get_optimum()
    x_gr = gr_solver.get_optimizer()
    y_gr = gr_solver.get_optimum()

    # Flatten optimizers to 1D vectors (len = 2)
    x_blk = np.asarray(x_blk).reshape(-1)
    x_gr = np.asarray(x_gr).reshape(-1)

    # Feasibility
    assert prob06_cons(x_blk) == 1.0
    assert prob06_cons(x_gr) == 1.0

    # Both are solving the same underlying problem; solutions should be similar.
    assert abs(y_blk - y_gr) < 0.1


def test_tree_root_bounds_match_initial_domain(quadratic_solver):
    """
    The tutorial accesses `bcp_solver.Tree[0][0]` and plots bounds.
    This test checks that the root node spans [-1, 1] x [-1, 1].
    """
    tree = quadratic_solver.Tree
    assert 0 in tree
    assert 0 in tree[0]

    root = tree[0][0]
    bounds = root.bounds  # shape (2, dim): [lower, upper]

    # Lower bounds for x0, x1
    np.testing.assert_allclose(bounds[0], np.array([-1.0, -1.0]), atol=1e-8)
    # Upper bounds for x0, x1
    np.testing.assert_allclose(bounds[1], np.array([1.0, 1.0]), atol=1e-8)


def test_tree_is_nonempty_with_multiple_levels(quadratic_solver):
    """
    After running DDSBB, the Tree used for branch-and-bound should contain
    at least the root level and (typically) additional levels.
    """
    levels = list(quadratic_solver.Tree.keys())
    assert len(levels) >= 1
    assert 0 in levels

    # There should be at least one node on each level present
    for level in levels:
        assert len(quadratic_solver.Tree[level]) >= 1
