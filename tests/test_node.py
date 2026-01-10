# tests/test_node.py

import numpy as np
from PyDDSBB._node import Node


def test_node_initialization_sets_basic_attributes():
    bounds = np.array([
        [0.0, -1.0, 2.0],
        [1.0,  3.0, 5.0],
    ])
    level = 2
    node_id = 7
    parent_id = 3

    n = Node(level=level, node=node_id, bounds=bounds, pn=parent_id)

    # basic attributes
    assert n.level == level
    assert n.node == node_id
    assert n.pn == parent_id

    assert np.allclose(n.bounds, bounds)
    expected_xrange = bounds[1, :] - bounds[0, :]
    assert np.allclose(n.xrange, expected_xrange)

    assert n.min_xrange == max(expected_xrange)

    assert n.decision == 1
    assert isinstance(n.child, list)
    assert n.child == []


def test_add_child_appends_children_in_order():
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=0, node=0, bounds=bounds)

    n.add_child(1)
    n.add_child(2)
    n.add_child(5)

    assert n.child == [1, 2, 5]


def test_add_parent_overwrites_pn():
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=1, node=10, bounds=bounds, pn=None)

    assert n.pn is None
    n.add_parent(4)
    assert n.pn == 4

    n.add_parent(9)
    assert n.pn == 9


def test_add_data_sets_x_and_y():
    bounds = np.array([[0.0, 0.0],
                       [1.0, 1.0]])
    n = Node(level=0, node=0, bounds=bounds)

    X = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6]])
    Y = np.array([1.0, 2.0, 3.0])

    n.add_data(X, Y)

    assert hasattr(n, "x")
    assert hasattr(n, "y")
    assert n.x.shape == X.shape
    assert n.y.shape == Y.shape
    assert np.allclose(n.x, X)
    assert np.allclose(n.y, Y)


def test_add_score_sets_score():
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=0, node=0, bounds=bounds)

    score = np.array([0.1, 0.5, 0.2])
    n.add_score(score)

    assert hasattr(n, "score")
    assert n.score.shape == score.shape
    assert np.allclose(n.score, score)


def test_set_opt_local_and_set_opt_flb_and_set_lipschitz():
    bounds = np.array([[0.0, 0.0],
                       [1.0, 2.0]])
    n = Node(level=0, node=0, bounds=bounds)

    fub = 1.234
    xopt = np.array([0.25, 1.75])
    n.set_opt_local(fub=fub, xopt=xopt)

    assert hasattr(n, "yopt_local")
    assert hasattr(n, "xopt_local")
    assert n.yopt_local == fub
    assert np.allclose(n.xopt_local, xopt)

    flb = 0.9
    n.set_opt_flb(flb)
    assert hasattr(n, "flb")
    assert n.flb == flb

    lipschitz = 12.5
    n.set_lipschitz(lipschitz)
    assert hasattr(n, "lipschitz")
    assert n.lipschitz == lipschitz


def test_set_decision_flag():
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=0, node=0, bounds=bounds)

    # default is 1 (active)
    assert n.decision == 1

    n.set_decision(0)
    assert n.decision == 0

    n.set_decision(1)
    assert n.decision == 1


def test_add_label_and_valid_ind():
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=0, node=0, bounds=bounds)

    # labels for 4 samples
    label = np.array([1, 0, 1, 1])
    valid_ind = [0, 2, 3]

    n.add_label(label)
    n.add_valid_ind(valid_ind)

    assert hasattr(n, "label")
    assert hasattr(n, "valid_ind")

    assert np.array_equal(n.label, label)
    assert n.valid_ind == valid_ind


def test_print_node_active_and_pruned(capsys):
    bounds = np.array([[0.0], [1.0]])
    n = Node(level=1, node=42, bounds=bounds)

    n.set_opt_local(fub=2.0, xopt=np.array([0.5]))
    n.set_opt_flb(flb=1.5)

    # Case 1: active node
    n.set_decision(1)
    n.print_node()
    captured = capsys.readouterr()
    out = captured.out

    assert "Node level:   1" in out
    assert "Node number:  42" in out
    assert "Local upper bound:  2.0" in out
    assert "Local lower bound:  1.5" in out
    assert "Local gap: 0.5" in out
    assert "Node active" in out

    # Case 2: pruned node
    n.set_decision(0)
    n.print_node()
    captured = capsys.readouterr()
    out2 = captured.out

    assert "Node pruned" in out2
