# tests/test_utilis.py

import numpy as np
from PyDDSBB._utilis import LHS


# -----------------------------------------------------------------------------
# initial_sample tests
# -----------------------------------------------------------------------------

def test_initial_sample_shape_and_bounds():
    np.random.seed(0)
    dim = 3
    n = 10
    X = LHS.initial_sample(dim, n)

    assert isinstance(X, np.ndarray)
    assert X.shape == (n, dim)

    assert np.all(X >= 0.0)
    assert np.all(X <= 1.0)


def test_initial_sample_no_duplicate_rows_small_dim():
    np.random.seed(1)
    dim = 2
    n = 15
    X = LHS.initial_sample(dim, n)

    X_round = np.round(X, 3)
    row_tuples = [tuple(row) for row in X_round]
    assert len(set(row_tuples)) == n


def test_initial_sample_reproducible_with_global_seed():
    """
    With the same global numpy seed, initial_sample should be deterministic.
    """
    dim = 3
    n = 25

    np.random.seed(123)
    X1 = LHS.initial_sample(dim, n)

    np.random.seed(123)
    X2 = LHS.initial_sample(dim, n)

    assert np.allclose(X1, X2)


def test_initial_sample_column_statistics_roughly_uniform():
    """
    For a reasonably large number of points, the columns should look roughly
    uniform: mean ~ 0.5 and variance ~ 1/12.
    """
    np.random.seed(5)
    dim = 4
    n = 200
    X = LHS.initial_sample(dim=dim, number_new_points=n)

    # mean around 0.5 with a loose tolerance
    means = X.mean(axis=0)
    assert np.all(means > 0.4)
    assert np.all(means < 0.6)

    # variance around 1/12 ~ 0.0833 with a loose tolerance
    vars_ = X.var(axis=0)
    assert np.all(vars_ > 0.05)
    assert np.all(vars_ < 0.12)


def test_initial_sample_bins_not_empty_in_each_dim():
    """
    For n >= 20, if we split [0,1] into 10 equal-width bins, each bin should
    have at least one point in most dimensions. 
    """
    np.random.seed(6)
    dim = 3
    n = 100
    X = LHS.initial_sample(dim=dim, number_new_points=n)

    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for j in range(dim):
        col = X[:, j]
        counts, _ = np.histogram(col, bins=bin_edges)
        # at least 7 out of 10 bins non-empty for each dim
        non_empty = np.sum(counts > 0)
        assert non_empty >= 7


# -----------------------------------------------------------------------------
# augmentLHS tests
# -----------------------------------------------------------------------------

def test_augmentlhs_shape_and_bounds():
    np.random.seed(4)
    original = np.array([[0.1, 0.2],
                         [0.8, 0.9]])
    k = 5
    new = LHS.augmentLHS(original, k)

    assert isinstance(new, np.ndarray)
    assert new.shape == (k, original.shape[1])

    # all new points inside [0,1]
    assert np.all(new >= 0.0)
    assert np.all(new <= 1.0)


def test_augmentlhs_no_exact_duplicates_with_original():
    np.random.seed(5)
    original = np.array([[0.1, 0.2],
                         [0.4, 0.6],
                         [0.8, 0.9]])
    k = 7
    new = LHS.augmentLHS(original, k)

    orig_round = np.round(original, 5)
    new_round = np.round(new, 5)

    for row in new_round:
        # no row of 'new' should exactly match a row of 'original' (up to rounding)
        assert not np.any(np.all(orig_round == row, axis=1))


def test_augmentlhs_all_points_still_in_unit_cube_after_combining():
    np.random.seed(6)
    dim = 3
    original = LHS.initial_sample(dim=dim, number_new_points=4)
    k = 6
    new = LHS.augmentLHS(original, k)

    all_points = np.vstack([original, new])
    assert all_points.shape == (4 + 6, dim)
    assert np.all(all_points >= 0.0)
    assert np.all(all_points <= 1.0)


def test_augmentlhs_does_not_modify_original_in_place():
    """
    Ensure the input array is not modified by augmentLHS.
    """
    np.random.seed(7)
    original = np.array([[0.1, 0.2],
                         [0.3, 0.4],
                         [0.7, 0.9]])
    original_copy = original.copy()

    _ = LHS.augmentLHS(original, 5)

    # original should remain unchanged
    assert np.allclose(original, original_copy)


def test_augmentlhs_reproducible_with_global_seed():
    np.random.seed(8)
    original = LHS.initial_sample(dim=2, number_new_points=5)

    np.random.seed(42)
    new1 = LHS.augmentLHS(original, 7)

    np.random.seed(42)
    new2 = LHS.augmentLHS(original, 7)

    assert np.allclose(new1, new2)


def test_augmentlhs_new_points_not_identical_and_some_separation_in_1d():
    """
    In 1D, check:
    - new points are not exactly equal to original points
    - there is at least a small positive distance from new to nearest original.
    """
    np.random.seed(9)
    dim = 1
    n_orig = 5

    original = LHS.initial_sample(dim=dim, number_new_points=n_orig)
    original_sorted = np.sort(original[:, 0])

    k = 5
    new = LHS.augmentLHS(original, k)
    new_vals = new[:, 0]

    # no exact duplicates with original (up to tight rounding)
    orig_round = np.round(original_sorted, 6)
    new_round = np.round(new_vals, 6)
    for v in new_round:
        assert v not in orig_round

    # distances from each new point to nearest original point
    dists = []
    for v in new_vals:
        dists.append(np.min(np.abs(original_sorted - v)))
    dists = np.array(dists)

    # at least some positive separation (no collision)
    assert np.min(dists) > 0.0
