# tests/test_splitter.py

import numpy as np

from PyDDSBB._splitter import Splitter
from PyDDSBB._node import Node


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_parent_node_1d(labels_mixed=True):
    """
    Create a 1D parent node with bounds [0,1],
    samples on a grid, and feasibility labels.
    """
    bounds = np.array([[0.0],
                       [1.0]])
    parent = Node(level=0, node=0, bounds=bounds)

    x = np.linspace(0.0, 1.0, 11)[:, None]
    if labels_mixed:
        label = (x[:, 0] <= 0.5).astype(int)
    else:
        label = np.ones_like(x[:, 0], dtype=int)

    parent.add_data(x, np.zeros_like(x[:, 0]))
    parent.add_label(label)
    parent.add_valid_ind(list(range(len(x))))
    return parent


def make_parent_node_2d(labels_mixed=False):
    """
    2D node with bounds [0,1]x[0,2].
    If labels_mixed=True, label = 1 if x1 <= 0.5 else 0.
    """
    bounds = np.array([[0.0, 0.0],
                       [1.0, 2.0]])
    parent = Node(level=0, node=0, bounds=bounds)

    xs = []
    ys = []
    labels = []
    for x1 in np.linspace(0.0, 1.0, 5):
        for x2 in np.linspace(0.0, 2.0, 5):
            xs.append([x1, x2])
            ys.append(x1 + 0.1 * x2)
            if labels_mixed:
                labels.append(1 if x1 <= 0.5 else 0)
            else:
                labels.append(1)

    X = np.array(xs)
    Y = np.array(ys)
    labels = np.array(labels, dtype=int)

    parent.add_data(X, Y)
    parent.add_valid_ind(list(range(len(X))))
    parent.add_label(labels)
    return parent


# -----------------------------------------------------------------------------
# equal_bisection & longest_side
# -----------------------------------------------------------------------------

def test_equal_bisection_center():
    bounds = np.array([[0.0, -1.0],
                       [1.0,  3.0]])
    parent = Node(level=0, node=0, bounds=bounds)

    mid0 = Splitter.equal_bisection(parent, split_x=0)
    mid1 = Splitter.equal_bisection(parent, split_x=1)

    assert mid0 == 0.5 * (bounds[1, 0] - bounds[0, 0]) + bounds[0, 0]
    assert mid1 == 0.5 * (bounds[1, 1] - bounds[0, 1]) + bounds[0, 1]
    assert mid1 == 1.0


def test_longest_side_index():
    bounds = np.array([[0.0, -1.0, -0.5],
                       [0.1,  3.0,  0.0]])
    parent = Node(level=0, node=0, bounds=bounds)

    idx = Splitter.longest_side(parent)
    assert idx == 1


# -----------------------------------------------------------------------------
# split_reg behavior
# -----------------------------------------------------------------------------

def test_split_reg_splits_on_longest_side():
    s = Splitter(split_method="equal_bisection", variable_selection="longest_side", minimum_bd=0.05)
    parent = make_parent_node_2d()

    child1_bounds, child2_bounds = s.split(parent)

    # Should split along dimension with largest range: [0,2] -> dim 1
    assert child1_bounds.shape == parent.bounds.shape
    assert child2_bounds.shape == parent.bounds.shape

    # other dimension (0) unchanged
    assert np.allclose(child1_bounds[:, 0], parent.bounds[:, 0])
    assert np.allclose(child2_bounds[:, 0], parent.bounds[:, 0])

    # child1 is the RIGHT interval, child2 is the LEFT interval
    mid = Splitter.equal_bisection(parent, split_x=1)

    # child1: [mid, upper]
    assert np.isclose(child1_bounds[0, 1], mid)
    assert np.isclose(child1_bounds[1, 1], parent.bounds[1, 1])

    # child2: [lower, mid]
    assert np.isclose(child2_bounds[0, 1], parent.bounds[0, 1])
    assert np.isclose(child2_bounds[1, 1], mid)


# -----------------------------------------------------------------------------
# location_based (gini/purity) behavior
# -----------------------------------------------------------------------------

def test_location_based_gini_on_mixed_labels():
    parent = make_parent_node_1d(labels_mixed=True)
    s = Splitter(split_method="gini", variable_selection="gini", minimum_bd=0.05)

    child1_bounds, child2_bounds = s.split(parent)

    # child2 is left interval, child1 is right interval
    # Bounds should still cover [0,1] with a split strictly inside.
    split_point = child1_bounds[0, 0]  # lower bound of right child

    # Right child upper bound is 1.0
    assert np.isclose(child1_bounds[1, 0], 1.0)
    # Left child lower bound is 0.0
    assert np.isclose(child2_bounds[0, 0], 0.0)

    # Split strictly inside
    assert 0.0 < split_point < 1.0

    # Meeting point
    assert np.isclose(child2_bounds[1, 0], split_point)


def test_location_based_purity_on_mixed_labels():
    parent = make_parent_node_1d(labels_mixed=True)
    s = Splitter(split_method="purity", variable_selection="purity", minimum_bd=0.05)

    child1_bounds, child2_bounds = s.split(parent)

    split_point = child1_bounds[0, 0]

    assert np.isclose(child1_bounds[1, 0], 1.0)
    assert np.isclose(child2_bounds[0, 0], 0.0)
    assert 0.0 < split_point < 1.0
    assert np.isclose(child2_bounds[1, 0], split_point)


def test_location_based_fallback_all_feasible():
    """
    When all labels are the same, location_based should fall back to
    longest_side + equal_bisection, using the same child1/right, child2/left
    convention.
    """
    parent = make_parent_node_1d(labels_mixed=False)
    s = Splitter(split_method="gini", variable_selection="gini", minimum_bd=0.05)

    # split via location_based
    child1_bounds, child2_bounds = s.split(parent)

    split_x = Splitter.longest_side(parent)
    mid = Splitter.equal_bisection(parent, split_x)

    expected_child1 = parent.bounds.copy()
    expected_child2 = parent.bounds.copy()

    # child1: [mid, upper], child2: [lower, mid]
    expected_child1[0, split_x] = mid
    expected_child2[1, split_x] = mid

    assert np.allclose(child1_bounds, expected_child1)
    assert np.allclose(child2_bounds, expected_child2)


# -----------------------------------------------------------------------------
# gini and purity minimum_bd behavior
# -----------------------------------------------------------------------------

def test_gini_returns_none_when_range_too_small():
    # bounds width < minimum_bd so gini should return (None, 2.)
    bounds = np.array([[0.0],
                       [0.01]])  
    parent = Node(level=0, node=0, bounds=bounds)

    x = np.linspace(0.0, 0.01, 5)[:, None]
    label = np.array([1, 0, 1, 0, 1], dtype=int)

    parent.add_data(x, np.zeros_like(x[:, 0]))
    parent.add_label(label)
    parent.add_valid_ind(list(range(len(x))))

    s = Splitter(split_method="gini", variable_selection="gini", minimum_bd=0.05)
    loc, score = s.gini(parent, split_x=0)

    assert loc is None
    assert score == 2.0


def test_purity_returns_none_when_range_too_small():
    bounds = np.array([[0.0],
                       [0.01]])
    parent = Node(level=0, node=0, bounds=bounds)

    x = np.linspace(0.0, 0.01, 5)[:, None]
    label = np.array([1, 0, 1, 0, 1], dtype=int)

    parent.add_data(x, np.zeros_like(x[:, 0]))
    parent.add_label(label)
    parent.add_valid_ind(list(range(len(x))))

    s = Splitter(split_method="purity", variable_selection="purity", minimum_bd=0.05)
    loc, score = s.purity(parent, split_x=0)

    assert loc is None
    assert score == 2.0


# -----------------------------------------------------------------------------
# hybrid behavior
# -----------------------------------------------------------------------------

def test_hybrid_mixed_labels_splits_inside():
    """
    With mixed labels, hybrid should:
    - choose a split dimension using variable_selection (svr_var_select)
    - choose a split location using criteria (gini/purity)
    - create two children covering the full interval on that dimension.
    """
    parent = make_parent_node_2d(labels_mixed=True)
    s = Splitter(split_method="gini", variable_selection="svr_var_select", minimum_bd=0.05)

    child1_bounds, child2_bounds = s.split(parent)

    # children must have same shape as parent bounds
    assert child1_bounds.shape == parent.bounds.shape
    assert child2_bounds.shape == parent.bounds.shape

    # find split dimension as one where bounds differ from parent
    diffs = np.any(child1_bounds != parent.bounds, axis=0) | np.any(child2_bounds != parent.bounds, axis=0)
    split_dims = np.where(diffs)[0]
    assert len(split_dims) == 1
    split_x = split_dims[0]

    #child1 is right interval, child2 is left interval
    split_point = child1_bounds[0, split_x]
    assert np.isclose(child1_bounds[1, split_x], parent.bounds[1, split_x])
    assert np.isclose(child2_bounds[0, split_x], parent.bounds[0, split_x])
    assert np.isclose(child2_bounds[1, split_x], split_point)
    assert parent.bounds[0, split_x] < split_point < parent.bounds[1, split_x]


def test_hybrid_all_feasible_fallback_equal_bisection():
    """
    When all labels are 1 or all 0, hybrid should use equal_bisection
    on the dimension selected by variable_selection.
    """
    parent = make_parent_node_2d(labels_mixed=False)
    s = Splitter(split_method="gini", variable_selection="svr_var_select", minimum_bd=0.05)

    # variable_selection dimension
    split_x = s.svr_var_select(parent)
    mid = Splitter.equal_bisection(parent, split_x)

    child1_bounds, child2_bounds = s.split(parent)

    expected_child1 = parent.bounds.copy()
    expected_child2 = parent.bounds.copy()
    # child1: [mid, upper], child2: [lower, mid]
    expected_child1[0, split_x] = mid
    expected_child2[1, split_x] = mid

    assert np.allclose(child1_bounds, expected_child1)
    assert np.allclose(child2_bounds, expected_child2)


# -----------------------------------------------------------------------------
# svr_var_select
# -----------------------------------------------------------------------------

def test_svr_var_select_sets_score_and_returns_valid_dim():
    parent = make_parent_node_2d()
    s = Splitter(split_method="gini", variable_selection="svr_var_select", minimum_bd=0.05)

    idx = s.svr_var_select(parent)

    # index must be a valid dimension index
    assert idx in [0, 1]

    # score must exist and match dimension count
    assert hasattr(parent, "score")
    assert parent.score.shape == (parent.x.shape[1],)


def test_svr_var_select_reuses_existing_score():
    parent = make_parent_node_2d()
    s = Splitter(split_method="gini", variable_selection="svr_var_select", minimum_bd=0.05)

    # First call creates score
    idx1 = s.svr_var_select(parent)
    score_before = parent.score.copy()

    # Second call should reuse existing parent.score
    idx2 = s.svr_var_select(parent)
    score_after = parent.score.copy()

    assert idx1 in [0, 1]
    assert idx2 in [0, 1]
    # scores unchanged between calls
    assert np.allclose(score_before, score_after)
